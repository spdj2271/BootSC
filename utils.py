# -*- coding: utf-8 -*-
from lightly.models.utils import normalize_weight
from sklearn import metrics
import numpy as np
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch
import torchvision
from lightly.data import LightlyDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import imagenet_10_cc, imagenet_dogs_cc, Self_CIFAR10, Self_CIFAR100, _cifar100_class_to_superclass
from typing import List, Union


def get_data_loaders(batch_size: int, dataset_train_ssl, dataset_train_kNN, dataset_test, num_workers):
    # Helper method to create dataloaders
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


def sinkhorn(out: torch.Tensor, iterations: int = 3, epsilon: float = 0.05) -> torch.Tensor:
    # Sinkhorn’s Fixed Point Iteration
    Q = torch.exp(out / epsilon).t()
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    B = Q.shape[1]
    for _ in range(iterations):
        # normalize rows \beta
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        # normalize columns \alpha
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()


def get_ds_loaders(dataset_name, simclr_transform, test_transforms, batch_size, num_workers):
    # configure datasets according the given 'dataset_name'
    if dataset_name == 'cifar10':
        base_train = Self_CIFAR10(root='./datasets/cifar10')
        base_test = Self_CIFAR10(root='./datasets/cifar10')
        classes = 10
    elif dataset_name == 'cifar100':
        base_train = Self_CIFAR100(root='./datasets/cifar10')
        base_train = _cifar100_class_to_superclass(base_train)
        base_test = Self_CIFAR100(root='./datasets/cifar10')
        base_test = _cifar100_class_to_superclass(base_test)
        classes = 20
    elif dataset_name == 'imagenet10':
        path = './datasets/imagenet/imagenet10'
        base_train, base_test = imagenet_10_cc(path)
        classes = 10
    elif dataset_name == 'imagenet_dogs':
        path = './datasets/imagenet/imagenet_dogs'
        base_train, base_test = imagenet_dogs_cc(path)
        classes = 15
    elif dataset_name == 'tiny_imagenet':
        path = './datasets/imagenet/tiny_imagenet'
        base_train = torchvision.datasets.ImageFolder(root=path)
        base_test = torchvision.datasets.ImageFolder(root=path)
        classes = 200
    else:
        raise Exception(f"unknown dataset_name={dataset_name}")
    # Creating a LightlyDataset for training and test data
    dataset_train_ssl = LightlyDataset.from_torch_dataset(base_train, transform=simclr_transform)
    dataset_test = LightlyDataset.from_torch_dataset(base_test, transform=test_transforms)
    # Creating a DataLoader for the training and test data
    dataloader_train_ssl = torch.utils.data.DataLoader(dataset_train_ssl, batch_size=batch_size, shuffle=True,
                                                       drop_last=True, persistent_workers=True, pin_memory=True,
                                                       num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                                  drop_last=False, persistent_workers=True, pin_memory=True,
                                                  num_workers=num_workers)
    return dataloader_train_ssl, dataloader_test, classes


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Solve the The best mapping between predicted cluster assignments and ground truth labels
    num_samples = flat_targets.shape[0]
    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))
    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res


@torch.no_grad()
def hungarian_evaluate(predictions, targets, num_classes=None):
    # Compute clustering performance metric ACC, NMI, ARI
    if num_classes is None:
        num_classes = len(torch.unique(targets))
    num_elems = predictions.shape[0]
    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    return acc, nmi, ari


class CustomCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    # lr restarts;
    # This custom class adds functionality to modify the learning rate at each restart of the cosine annealing cycle.
    def __init__(self, optimizer, T_0, lr_factor=0.1):
        super(CustomCosineAnnealingWarmRestarts, self).__init__(optimizer, T_0)
        self.lr_factor = lr_factor
        self.first_run = True

    def get_lr(self):
        if self.T_cur == 0 and not self.first_run:
            self.base_lrs[0] *= self.lr_factor
        self.first_run = False
        return super(CustomCosineAnnealingWarmRestarts, self).get_lr()


class Prototypes(nn.Module):
    # online clustering layer
    def __init__(self, input_dim: int = 128, n_prototypes: Union[List[int], int] = 3000,
                 n_steps_frozen_prototypes: int = 0):
        super(Prototypes, self).__init__()
        self.n_prototypes = (
            n_prototypes if isinstance(n_prototypes, list) else [n_prototypes]
        )
        self._is_single_prototype = True if isinstance(n_prototypes, int) else False
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, prototypes) for prototypes in self.n_prototypes]
        )
        self.n_steps_frozen_prototypes = n_steps_frozen_prototypes

    def forward(self, x, step=None) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._freeze_prototypes_if_required(step)
        out = []
        for layer in self.heads:
            out.append(layer(x))
        return out[0] if self._is_single_prototype else out

    def normalize(self):
        """Normalizes the prototypes so that they are on the unit sphere."""
        for layer in self.heads:
            normalize_weight(layer.weight)

    def _freeze_prototypes_if_required(self, step):
        if self.n_steps_frozen_prototypes > 0:
            if step is None:
                raise ValueError(
                    "`n_steps_frozen_prototypes` is greater than 0, please"
                    " provide the `step` argument to the `forward()` method."
                )
            self.requires_grad_(step >= self.n_steps_frozen_prototypes)
