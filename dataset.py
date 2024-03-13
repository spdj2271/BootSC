# -*- coding: utf-8 -*-
import glob
import os
from scipy import io as sio
import torch
from torch.utils.data import Dataset
import collections.abc
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional
import pickle
import os.path


class ImageDataset(Dataset):
    def __init__(self, data, annotations=None, transformations=None, data_reader=None):
        """
        Image dataset class
        :param data: The samples. Can be a matrix, list of paths to the images, or a data type compatible with the data_reader
        :param annotations: If not None it should be a list or vector with the corresponding labels for each sample
        :param data_reader: Optional, an object to read the data. Use for custom data structures (e.g. h5 files)
        """
        self.data_reader = data_reader or ImageReader(data, annotations)

    def __len__(self):
        return self.data_reader.__len__()

    def __getitem__(self, index):
        image, annotation = self.data_reader.__getitem__(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, annotation

    def _get_data_type(self, data):
        data_type = None
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            data_type = "tensor"
        elif isinstance(data, collections.abc.Sequence):
            if isinstance(data[0], str):
                data_type = "files"
        assert data_type is not None

        return data_type


class ImageReader:
    def __init__(self, data, annotations=None):
        self.data = data
        self.annotations = annotations
        self.tensor2PIL = T.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, torch.Tensor):
            image = self.tensor2PIL(image)
        else:
            image = Image.fromarray(image)

        if self.annotations is not None:
            annotation = int(self.annotations[index])
        else:
            annotation = None

        return image, annotation


def load_image(path):
    """Loads an image from given path"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


### ImageNet ###
def get_imagenet(dataset_path=None, version="default"):
    if dataset_path is None:
        dataset_path = "./datasets/ImageNet"
    if os.path.isdir(dataset_path + '/ILSVRC2012_devkit_t12') and os.path.isdir(
            dataset_path + '/val') and os.path.isdir(dataset_path + '/train'):
        return get_imagenet_file_reading(dataset_path, version)
    elif os.path.isfile(dataset_path + '/train.h5') and os.path.isfile(dataset_path + '/val.h5'):
        return dataset_path + '/train.h5', None, dataset_path + '/val.h5', None


def get_imagenet_file_reading(dataset_path, version):
    idx_to_wnid, wnid_to_idx, wnid_to_classes = _parse_meta_mat(dataset_path + '/ILSVRC2012_devkit_t12')
    if version != "default":
        if version == "imagenet-dogs":
            c_folder = ["n02085936", "n02086646", "n02088238", "n02091467", "n02097130", "n02099601", "n02101388",
                        "n02101556", "n02102177", "n02105056", "n02105412", "n02105855", "n02107142", "n02110958",
                        "n02112137"]
        elif version == "imagenet-10":
            c_folder = ["n02056570", "n02085936", "n02128757", "n02690373", "n02692877", "n03095699", "n04254680",
                        "n04285008", "n04467665", "n07747607"]
        elif version == 'imagenet-tiny':
            c_folder = ['n02795169', 'n02769748', 'n07920052', 'n02917067', 'n01629819', 'n02058221', 'n02793495',
                        'n04251144', 'n02814533', 'n02837789', 'n01770393', 'n01910747', 'n03649909', 'n02124075',
                        'n01774750', 'n06596364', 'n03838899', 'n02480495', 'n09256479', 'n03085013', 'n01443537',
                        'n04376876', 'n03404251', 'n03930313', 'n03089624', 'n04371430', 'n04254777', 'n02909870',
                        'n07614500', 'n02977058', 'n04259630', 'n07579787', 'n02950826', 'n02279972', 'n03424325',
                        'n03854065', 'n02403003', 'n01742172', 'n01882714', 'n03977966', 'n02669723', 'n02226429',
                        'n04366367', 'n02002724', 'n03891332', 'n01768244', 'n02509815', 'n03544143', 'n02321529',
                        'n02099601', 'n02948072', 'n04456115', 'n02236044', 'n03126707', 'n02074367', 'n03255030',
                        'n01950731', 'n02268443', 'n04501370', 'n03970156', 'n04099969', 'n04023962', 'n02085620',
                        'n02823428', 'n04265275', 'n02113799', 'n01784675', 'n03706229', 'n03100240', 'n04532106',
                        'n02788148', 'n07753592', 'n03983396', 'n04399382', 'n03902125', 'n02814860', 'n03014705',
                        'n09428293', 'n02481823', 'n04597913', 'n01944390', 'n03355925', 'n07871810', 'n03042490',
                        'n02190166', 'n04486054', 'n04008634', 'n02906734', 'n02699494', 'n04070727', 'n01855672',
                        'n09246464', 'n02364673', 'n07768694', 'n02883205', 'n04532670', 'n02815834', 'n02165456',
                        'n04540053', 'n02802426', 'n04356056', 'n03670208', 'n04562935', 'n01641577', 'n07615774',
                        'n07734744', 'n03584254', 'n01698640', 'n04507155', 'n02125311', 'n03179701', 'n07873807',
                        'n04179913', 'n04560804', 'n03393912', 'n02841315', 'n02843684', 'n09193705', 'n02437312',
                        'n04275548', 'n04118538', 'n02099712', 'n07747607', 'n03250847', 'n04133789', 'n02094433',
                        'n04074963', 'n02129165', 'n03637318', 'n02056570', 'n02410509', 'n03980874', 'n03400231',
                        'n03814639', 'n03026506', 'n01644900', 'n04398044', 'n02666196', 'n03444034', 'n04487081',
                        'n02486410', 'n02808440', 'n04149813', 'n12267677', 'n03662601', 'n02233338', 'n07711569',
                        'n02791270', 'n04465501', 'n03599486', 'n07720875', 'n03447447', 'n03804744', 'n04311004',
                        'n07695742', 'n07583066', 'n07715103', 'n04328186', 'n01917289', 'n02106662', 'n02927161',
                        'n02395406', 'n02231487', 'n02123394', 'n03976657', 'n02423022', 'n03770439', 'n04067472',
                        'n02206856', 'n04285008', 'n03617480', 'n03733131', 'n02415577', 'n04146614', 'n03388043',
                        'n01945685', 'n02892201', 'n03160309', 'n02281406', 'n02999410', 'n02504458', 'n04596742',
                        'n02132136', 'n03763968', 'n03796401', 'n07875152', 'n01983481', 'n07749582', 'n01774384',
                        'n03201208', 'n01984695', 'n02963159', 'n02123045', 'n09332890', 'n03992509', 'n02988304',
                        'n04417672', 'n02730930', 'n03937543', 'n03837869']
        idx_to_wnid_, wnid_to_idx_, wnid_to_classes_ = {}, {}, {}
        for i, c_ in enumerate(c_folder):
            idx_to_wnid_[i] = c_
            wnid_to_idx_[c_] = i
            wnid_to_classes_[c_] = wnid_to_classes[c_]
        idx_to_wnid, wnid_to_idx, wnid_to_classes = idx_to_wnid_, wnid_to_idx_, wnid_to_classes_
    train_path = dataset_path + '/train/'
    train_samples, train_labels = [], []
    for k in wnid_to_classes.keys():
        k_samples = glob.glob(train_path + k + '/*')
        train_samples += k_samples
        train_labels += len(k_samples) * [wnid_to_idx[k]]

    val_labels = _parse_val_groundtruth_txt(dataset_path + '/ILSVRC2012_devkit_t12')
    val_path = dataset_path + '/val'
    val_samples = glob.glob(val_path + '/*')
    val_samples.sort()

    return np.array(train_samples), np.array(train_labels), np.array(val_samples), np.array(val_labels)


def _parse_val_groundtruth_txt(devkit_root):
    file = os.path.join(devkit_root, "data",
                        "ILSVRC2012_validation_ground_truth.txt")
    with open(file, 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) - 1 for val_idx in val_idcs]


def _parse_meta_mat(devkit_root):
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx - 1: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_idx = {wnid: idx - 1 for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}

    return idx_to_wnid, wnid_to_idx, wnid_to_classes


def _get_cc_datasets(data, labels, crop_size, blur=False, color_jitter_s=0.5, data_val=None,
                     labels_val=None):
    if data_val is None:
        data_val, labels_val = data, labels
    train_dataset = ImageDataset(data, annotations=labels)
    val_dataset = ImageDataset(data_val, annotations=labels_val)
    return train_dataset, val_dataset


def imagenet_10_cc(dataset_path=None, crop_size=224, *args, **kwargs):
    data, labels, _, _ = get_imagenet(dataset_path, "imagenet-10")
    train_dataset, val_dataset = _get_cc_datasets(data, labels, crop_size, blur=True, color_jitter_s=1)
    return train_dataset, val_dataset


def imagenet_dogs_cc(dataset_path=None, crop_size=224, *args, **kwargs):
    data, labels, _, _ = get_imagenet(dataset_path, "imagenet-dogs")
    train_dataset, val_dataset = _get_cc_datasets(data, labels, crop_size, blur=True, color_jitter_s=1)
    return train_dataset, val_dataset


def imagenet_tiny(dataset_path=None, crop_size=224, *args, **kwargs):
    data, labels, _, _ = get_imagenet(dataset_path, "imagenet-tiny")
    train_dataset, val_dataset = _get_cc_datasets(data, labels, crop_size, blur=True, color_jitter_s=1)
    return train_dataset, val_dataset


class Self_CIFAR10(CIFAR10):
    def __init__(self, root: str, train: str = 'all', transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download=True):
        super().__init__(root, transform=transform, target_transform=target_transform, download=True)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train == 'train':
            downloaded_list = self.train_list
        elif self.train == 'test':
            downloaded_list = self.test_list
        elif self.train == 'all':
            downloaded_list = self.train_list + self.test_list
        else:
            raise Exception(f"unknown self.train={self.train}")
        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()


class Self_CIFAR100(Self_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


def _cifar100_class_to_superclass(dataset):
    aquatic_mammals = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    fish = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    flowers = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
    food_containers = ['bottle', 'bowl', 'can', 'cup', 'plate']
    fruits_and_vegetables = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
    house_electrical_devices = ['clock', 'keyboard', 'lamp', 'telephone', 'television']
    household_furniture = ['couch', 'bed', 'chair', 'table', 'wardrobe']
    insects = ['bee', 'butterfly', 'beetle', 'caterpillar', 'cockroach']
    large_carnivores = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    large_man_made_outdoor_things = ['bridge', 'castle', 'house', 'road', 'skyscraper']
    large_natural_outdoor_scenes = ['cloud', 'forest', 'mountain', 'plain', 'sea']
    large_omnivores_herbivores = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
    medium_sized_mammals = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
    non_insect_invertebrates = ['crab', 'lobster', 'snail', 'spider', 'worm']
    people = ['baby', 'boy', 'girl', 'man', 'woman']
    reptiles = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
    small_mammals = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    trees = ['maple_tree', 'oak_tree', 'pine_tree', 'palm_tree', 'willow_tree']
    vehicles_1 = ['bicycle', 'bus', 'pickup_truck', 'motorcycle', 'train']
    vehicles_2 = ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

    superclasses_lists = [aquatic_mammals, fish, flowers, food_containers, fruits_and_vegetables,
                          house_electrical_devices, household_furniture, insects, large_carnivores,
                          large_man_made_outdoor_things, large_natural_outdoor_scenes, large_omnivores_herbivores,
                          medium_sized_mammals, non_insect_invertebrates, people, reptiles, small_mammals, trees,
                          vehicles_1, vehicles_2]
    superclasses_names = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                          'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                          'large man-made outdoor things', 'large natural outdoor scenes',
                          'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
                          'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

    class_2_superclass = {}
    for subclasses, superclass in zip(superclasses_lists, superclasses_names):
        for v in subclasses:
            class_2_superclass[v] = superclass

    class_to_idx = {}
    i = 0
    for k in class_2_superclass.values():
        if k not in class_to_idx.keys():
            class_to_idx[k] = i
            i += 1
    dataset_dict = dataset.class_to_idx
    idx_to_supclass_idx = {}
    for k, v in dataset_dict.items():
        idx_to_supclass_idx[v] = class_to_idx[class_2_superclass[k]]
    for i in range(len(dataset.targets)):
        dataset.targets[i] = idx_to_supclass_idx[dataset.targets[i]]
    return dataset
