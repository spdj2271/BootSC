# Bootstrap Deep Spectral Clustering with Optimal Transport (BootSC)

BootSC is an online deep spectral clustering network.

[Paper](https://arxiv.org/pdf/2508.04200)

## Which script should I use?
**Update:** April 30, 2026

Please choose the training script according to your input type:

- For **raw image input**, please use `BootSC.py`.
- For **pre-extracted feature vectors or non-image input**, please use `BootSC_for_feature_input.py`.

In short, if your input is an image tensor, use `BootSC.py`; if your input is already represented as a feature matrix `X` with shape `[N, D]`, use `BootSC_for_feature_input.py`.

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

## Citations
If you find our work helpful, your citations are highly appreciated:
```
@article{guo2026bootstrap,
  author={Guo, Wengang and Ye, Wei and Chen, Chunchun and Sun, Xin and Böhm, Christian and Plant, Claudia and Rahardja, Susanto},
  journal={IEEE Transactions on Multimedia}, 
  title={Bootstrap Deep Spectral Clustering With Optimal Transport}, 
  year={2026},
  volume={28},
  number={},
  pages={531-544},
  doi={10.1109/TMM.2025.3623492}
}
```
