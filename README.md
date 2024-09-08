# Neural Network Signal Processing on Torch

![Python version support](https://img.shields.io/pypi/pyversions/nnspt)
[![PyPI version](https://badge.fury.io/py/nnspt.svg)](https://badge.fury.io/py/nnspt)
[![Downloads](https://pepy.tech/badge/nnspt/month)](https://pepy.tech/project/nnspt?versions=0.0.*)

NNSPT is a Python library for neural network signal processing on PyTorch.

## Table of contents
- [Authors](#authors)
- [Installation](#installation)
- [A simple example](#a-simple-example)
- [Advanced examples](#advanced-examples)
- [Available components](#available-components)
- [Citing](#citing)

## Authors
[**Rostislav Epifanov** — Researcher in Novosibirsk]()

## Installation
Installation from PyPI:

```
pip install nnspt
```

Installation from GitHub:

```
pip install git+https://github.com/rostepifanov/nnspt
```

## A simple example
```python
from nnspt.segmentation.unet import Unet

model = Unet(encoder='tv-resnet34')
```

## Advanced examples

 - [Kaggle notebook](https://www.kaggle.com/code/rostepifanov/nnspt-1d-effitientnet-v2-b1-starter-lb-0-63) HMS EEG competition (EEG classification)
 - [Kaggle notebook](https://www.kaggle.com/code/rostepifanov/nnspt-eeg-1d-cnn) GAL EEG competition (EEG segmentation)
 - [Kaggle notebook](https://www.kaggle.com/code/rostepifanov/nnstp-emg-1d-cnn) Handwritten digit classification by EMG record (EMG classification)

## Available components
#### Encoders

  * <details> <summary>ResNet</summary>

    | Name         | Weights | Params |
    | :----        | :---:   | :---:  |
    | tv-resnet18  | -       | 3.8M   |
    | tv-resnet34  | -       | 7.2M   |
    | tv-resnet50  | -       | 15.9M  |
    | tv-resnet101 | -       | 28.2M  |
    | tv-resnet152 | -       | 38.4M  |
  </details>

  * <details> <summary>ResNeXt</summary>

    | Name                 | Weights | Params |
    | :----                | :---:   | :---:  |
    | tv-resnext50_32x4d   | -       | 22M    |
    | tv-resnext101_32x4d  | -       | 40.3M  |
    | tv-resnext101_32x8d  | -       | 79.6M  |
    | tv-resnext101_32x16d | -       | 163.5M |
    | tv-resnext101_32x32d | -       | 352.6M |
    | tv-resnext101_32x48d | -       | 570.1M |
  </details>

  * <details> <summary>DenseNet</summary>

    | Name           | Weights | Params |
    | :----          | :---:   | :---:  |
    | tv-densenet121 | -       | 5.5M   |
    | tv-densenet169 | -       | 10.4M  |
    | tv-densenet201 | -       | 15.6M  |
    | tv-densenet161 | -       | 22.1M  |
  </details>

  * <details> <summary>EfficientNetV1</summary>

    | Name                 | Weights | Params |
    | :----                | :---:   | :---:  |
    | timm-efficientnet-b0 | -       | 3.4M   |
    | timm-efficientnet-b1 | -       | 5.9M   |
    | timm-efficientnet-b2 | -       | 6.9M   |
    | timm-efficientnet-b3 | -       | 9.8M   |
    | timm-efficientnet-b4 | -       | 16.3M  |
    | timm-efficientnet-b5 | -       | 26.7M  |
    | timm-efficientnet-b6 | -       | 38.6M  |
    | timm-efficientnet-b7 | -       | 61.1M  |
    | timm-efficientnet-b8 | -       | 81.2M  |
    | timm-efficientnet-l2 | -       | 463.4M |
  </details>

  * <details> <summary>EfficientNetLite</summary>

    | Name                    | Weights | Params |
    | :----                   | :---:   | :---:  |
    | timm-efficientnet-lite0 | -       | 2.8M   |
    | timm-efficientnet-lite1 | -       | 3.5M   |
    | timm-efficientnet-lite2 | -       | 4.1M   |
    | timm-efficientnet-lite3 | -       | 6.1M   |
    | timm-efficientnet-lite4 | -       | 10.7M  |
  </details>

  * <details> <summary>EfficientNetV2</summary>

    | Name                   | Weights | Params |
    | :----                  | :---:   | :---:  |
    | timm-efficientnetv2-b0 | -       | 5.4M   |
    | timm-efficientnetv2-b1 | -       | 6.3M   |
    | timm-efficientnetv2-b2 | -       | 8.1M   |
    | timm-efficientnetv2-b3 | -       | 12.1M  |
    | timm-efficientnetv2-s  | -       | 19.1M  |
    | timm-efficientnetv2-m  | -       | 50.8M  |
    | timm-efficientnetv2-l  | -       | 113.6M |
    | timm-efficientnetv2-xl | -       | 202.5M |
  </details>

  * <details> <summary>MobileNetV2</summary>

    | Name                 | Weights | Params |
    | :----                | :---:   | :---:  |
    | timm-mobilenetv2-035 | -       | 0.2M   |
    | timm-mobilenetv2-050 | -       | 0.4M   |
    | timm-mobilenetv2-075 | -       | 1.1M   |
    | timm-mobilenetv2-100 | -       | 1.7M   |
    | timm-mobilenetv2-140 | -       | 3.4M   |

    NOTE: digits in name mean **channel_multiplier**
  </details>

  * <details> <summary>ConvNeXt</summary>

    | Name                  | Weights | Params |
    | :----                 | :---:   | :---:  |
    | timm-convnext-atto    | -       | 3.1M   |
    | timm-convnext-femto   | -       | 4.5M   |
    | timm-convnext-pico    | -       | 8.1M   |
    | timm-convnext-nano    | -       | 14.2M  |
    | timm-convnext-tiny    | -       | 26.7M  |
    | timm-convnext-small   | -       | 48.1M  |
    | timm-convnext-base    | -       | 85.4M  |
    | timm-convnext-large   | -       | 191.9M |
    | timm-convnext-xlarge  | -       | 341.1M |
    | timm-convnext-xxlarge | -       | -M     |
  </details>

  * <details> <summary>ConvNeXtV2</summary>

    | Name                    | Weights | Params |
    | :----                   | :---:   | :---:  |
    | timm-convnextv2-atto    | -       | 3.1M   |
    | timm-convnextv2-femto   | -       | 4.5M   |
    | timm-convnextv2-pico    | -       | 8.1M   |
  </details>

#### Pretraining

  * Autoencoder
  * BYOL [[paper](https://arxiv.org/abs/2006.07733)]

#### Segmentation

  * Unet [[paper](https://arxiv.org/abs/1505.04597)]
  * Unetpp [[paper](https://arxiv.org/abs/1807.10165)]

## Citing

If you find this library useful for your research, please consider citing:

```
@misc{epifanov2023nnspt,
  Author = {Rostislav Epifanov},
  Title = {NNSTP},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/rostepifanov/nnspt}}
}
```
