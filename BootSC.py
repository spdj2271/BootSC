# -*- coding: utf-8 -*-
import os
import numpy as np
# Importing PyTorch Lightning for high-level PyTorch framework functionality
import pytorch_lightning as pl
from lightly.transforms.utils import IMAGENET_NORMALIZE
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from lightly.models import ResNetGenerator
from lightly.models.modules import heads
from lightly.transforms import SimCLRTransform, BYOLTransform, BYOLView1Transform, BYOLView2Transform
# Importing pytorch
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# Importing utility functions and classes from 'utils.py'
from utils import hungarian_evaluate, get_ds_loaders, CustomCosineAnnealingWarmRestarts, Prototypes, sinkhorn


class Model(LightningModule):
    def __init__(self, dataloader_test, n_classes, max_epochs):
        super().__init__()
        # Storing parameters
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.dataloader_test = dataloader_test

        # Initializing the ResNet backbone, the projection head, online clustering layer
        resnet = ResNetGenerator("resnet-34")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1))
        self.projection_head = heads.SimCLRProjectionHead(512, 4096, 128)
        self.prototypes = Prototypes(128, self.n_classes)

        # Initializing the Learnable Temperature Parameter \tau
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(0.05))
        self.logit_scale_w = nn.Parameter(torch.ones([]) * np.log(0.05))

        # Placeholder for storing predictions, and targets during validation
        self.validation_step_features = []
        self.validation_step_prediction = []
        self.validation_step_targets = []

    def configure_optimizers(self):
        # Setting up the optimizer
        optim = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # Setting up the learning rate scheduler with lr restarts every 200 epochs
        scheduler = CustomCosineAnnealingWarmRestarts(optim, T_0=200, lr_factor=lr_factor)
        return [optim], [scheduler]

    def forward(self, x, is_orthonormal=False):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        # Orthonormal Re-parameterization Technique
        if is_orthonormal:
            U, _, Vh = torch.linalg.svd(z.detach(), full_matrices=False)
            z = z + (U @ Vh - z).detach()
        # \ell_2 normalize the embeddings
        z = F.normalize(z, dim=1, p=2)
        # compute cluster assignments
        p = self.prototypes(z)
        return z, p

    def training_step(self, batch, batch_idx):
        # \ell_2 normalize the prototypes of the online clustering layer
        self.prototypes.normalize()
        # Compute the Learnable Temperature Parameter \tau
        self.logit_scale.data = torch.clamp(self.logit_scale, max=0).data
        self.logit_scale_w.data = torch.clamp(self.logit_scale_w, max=0).data
        # Compute embeddings and cluster assignments
        (x0, x1), targets, _ = batch
        z0, p0 = self.forward(x0, is_orthonormal=True)
        z1, p1 = self.forward(x1, is_orthonormal=True)
        # Compute training loss
        loss, loss_graph, loss_cluster = self.criterion(z0, z1, p0, p1)
        return {"loss": loss, "loss_graph": loss_graph.item(), "loss_cluster": loss_cluster.item(),
                "logit_scale": self.logit_scale.exp().item(), "logit_scale_w": self.logit_scale_w.exp().item()}

    def criterion(self, z0: torch.Tensor, z1: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor):
        batch_size = z0.shape[0]
        # solve the optimal cluster assignments
        q0 = sinkhorn(p0.detach(), iterations=iterations, epsilon=epsilon)
        q1 = sinkhorn(p1.detach(), iterations=iterations, epsilon=epsilon)
        logit_scale = self.logit_scale.exp()
        # compute cluster loss
        loss_cluster = 0.5 * (self.subloss(p0, q1, logit_scale) + self.subloss(p1, q0, logit_scale))
        # off-diagonal affinity matrix
        p0_w = z0 @ z0.T
        p1_w = z1 @ z1.T
        diag_mask = torch.eye(batch_size, device=z0.device, dtype=torch.bool)
        p0_w = p0_w[~diag_mask].view(batch_size, -1)
        p1_w = p1_w[~diag_mask].view(batch_size, -1)
        # solve the optimal affinity matrix
        q0_w = sinkhorn(p0_w.detach(), iterations=iterations, epsilon=epsilon)
        q1_w = sinkhorn(p1_w.detach(), iterations=iterations, epsilon=epsilon)
        logit_scale_w = self.logit_scale_w.exp()
        # compute affinity loss
        loss_graph = 0.5 * (self.subloss(p0_w, q1_w, logit_scale_w) + self.subloss(p1_w, q0_w, logit_scale_w))
        #
        loss = loss_cluster * lamb + loss_graph
        return loss, loss_graph, loss_cluster

    def subloss(self, z: torch.Tensor, q: torch.Tensor, temperature=0.1):
        # Compute the cross-entropy loss with temperature
        return -torch.mean(torch.sum(q * F.log_softmax(z / temperature, dim=1), dim=1))

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # tensorboard log
        log_str = {"Loss/" + key: value for key, value in outputs.items()}  # prefix
        self.log("Loss/loss", value=log_str.pop('Loss/loss'), prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log_dict(log_str, prog_bar=False, on_epoch=True, batch_size=batch_size)
        self.log('Loss/lr', self.optimizers().param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        # Resetting storage lists at the beginning of validation
        if batch_idx == 0:
            self.validation_step_features.clear()
            self.validation_step_prediction.clear()
            self.validation_step_targets.clear()
        # Forward pass to get embeddings and cluster assignments
        img, target, _ = batch
        feature, prediction = self.forward(img, is_orthonormal=False)
        self.validation_step_prediction.append(prediction)
        self.validation_step_targets.append(target)

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.validation_step_prediction).detach().cpu().argmax(1)
        targets = torch.cat(self.validation_step_targets).detach().cpu()
        # Compute clustering performance metrics ACC, NMI, ARI
        acc_pred, nmi_pred, ari_pred = hungarian_evaluate(predictions, targets, self.n_classes)
        # tensorboard log
        metrics = {"acc_pred": acc_pred, "nmi_pred": nmi_pred, "ari_pred": ari_pred}
        metrics = {"Metric/" + key: value for key, value in metrics.items()}
        print(metrics)
        self.log_dict(metrics, prog_bar=False)



# Setting a global seed for reproducibility, Configuring GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
pl.seed_everything(seed=0)
torch.use_deterministic_algorithms(True)
torch.multiprocessing.set_sharing_strategy('file_system')
logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Configuring dataset and training hyperparameters
dataset_name = ['cifar10', 'cifar100', 'imagenet10', 'imagenet_dogs', 'tiny_imagenet'][0]
max_epochs = 1000
input_size = 32
batch_size = 256
num_workers = 8
lamb = 1.0
iterations = 3
epsilon = 0.05
lr_factor = 0.5

# Setting normalization parameters based on the selected dataset
mean = IMAGENET_NORMALIZE["mean"]
std = IMAGENET_NORMALIZE["std"]
if dataset_name == 'cifar10':
    batch_size = 256
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.24670, 0.2434, 0.2616]
    transform = BYOLTransform(
        BYOLView1Transform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std}),
        BYOLView2Transform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std}))
elif dataset_name == 'cifar100':
    mean = [0.5074, 0.4867, 0.4411]
    std = [0.2675, 0.2566, 0.2764]
    transform = BYOLTransform(
        BYOLView1Transform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std}),
        BYOLView2Transform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std}))
elif dataset_name == 'imagenet10':
    input_size = 64
    transform = SimCLRTransform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std})
elif dataset_name == 'imagenet_dogs':
    input_size = 64
    transform = SimCLRTransform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std})
elif dataset_name == 'tiny_imagenet':
    input_size = 32
    batch_size = 1024
    transform = SimCLRTransform(input_size=input_size, gaussian_blur=0.0, normalize={'mean': mean, 'std': std})
else:
    raise Exception()
test_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(input_size),
     torchvision.transforms.CenterCrop(input_size),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=mean, std=std)])

lr = 4e-2 * (batch_size / 256)

# Loading datasets
dataloader_train_ssl, dataloader_test, classes = get_ds_loaders(dataset_name, transform, test_transforms, batch_size,
                                                                num_workers)
# Initializing the BootSC model
BootSC = Model(dataloader_test, classes, max_epochs)
hyperparams = {var_name: var_value for var_name, var_value in locals().items()
               if not var_name.startswith("__") and type(var_value) in [bool, int, float, str, list]}
print(f'\033[32m{hyperparams}\033[0m')

# Setting up logging and checkpointing
logger = TensorBoardLogger(save_dir=os.path.join(logs_root_dir, dataset_name), name="")
logger.log_hyperparams(hyperparams)
checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"))

# Configuring and starting the training process
trainer = pl.Trainer(max_epochs=max_epochs, accelerator=accelerator, default_root_dir=logs_root_dir, logger=logger,
                     callbacks=[checkpoint_callback])
trainer.fit(BootSC, train_dataloaders=dataloader_train_ssl, val_dataloaders=dataloader_test)
