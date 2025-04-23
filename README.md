# Bootstrap Deep Spectral Clustering with Optimal Transport (BootSC)

BootSC is an online deep spectral clustering framework. This repository contains the PyTorch implementation of BootSC along with training scripts and utility functions.

## Usage

Clone the repository:

    git clone https://github.com/spdj2271/BootSC
    
Install dependencies:

    pip install -r requirements.txt

Data Preparation: 
Both the CIFAR-10 and CIFAR-100 datasets will be automatically downloaded and prepared during training. 
As for the ImageNet-10, ImageNet Dogs, and Tiny ImageNet datasets, please refer to [this link](https://github.com/XLearning-SCU/2021-AAAI-CC) for downloading and configuring.
By default, BootSC utilizes the CIFAR-10 dataset and you can alse choose specific datasets by modifying the `dataset_name` variable in the script (Line 147 in BootSC.py)：
````
dataset_name = ['cifar10', 'cifar100', 'imagenet10', 'imagenet_dogs', 'tiny_imagenet'][0] # for Cifar-10
````

Training:

    python BootSC.py

## Acknowledgements
We acknowledge the contributions of the PyTorch Lightning, Lightly, and torch communities for providing foundational libraries and resources used in this implementation.
