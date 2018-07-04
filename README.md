# Switchable Normalization

Switchable Normalization is a normalization technique that is able to learn different normalization operations for different normalization layers in a deep neural network in an end-to-end manner.

![](teaser.png?raw=true)

## Update

- 2018/7/2: The code of image classification and a pretrained model on ImageNet are released.
- More pretrained models and the code of object detection will be released soon.

## Introduction

This repository provides imagenet classification and object detection results and models trained with [Switchable Normalization](https://arxiv.org/abs/1806.10779):

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng},
  journal={arXiv:1806.10779},
  year={2018}
}
```
## Overview of Results

### Image Classification in ImageNet

**Comparisons of top-1 accuracies** on the validation set of ImageNet, by using ResNet50 trained with SN, BN, and GN in different batch size settings. The bracket (·, ·) denotes (#GPUs,#samples per GPU). In the bottom part, “GN-BN” indicates the difference between the accuracies of GN and BN. The “-” in (8, 1) of BN indicates it does not converge.
<table>
<tbody>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;">(8,32)</td>
<td style="text-align: left;">(8,16)</td>
<td style="text-align: left;">(8,8)</td>
<td style="text-align: left;">(8,4)</td>
<td style="text-align: left;">(8,2)</td>
<td style="text-align: left;">(1,16)</td>
<td style="text-align: left;">(1,32)</td>
<td style="text-align: left;">(8,1)</td>
<td style="text-align: left;">(1,8)</td>
</tr>
<tr class="even">
<td style="text-align: center;">BN <span class="citation" data-cites="BN"></span></td>
<td style="text-align: left;">76.4</td>
<td style="text-align: left;">76.3</td>
<td style="text-align: left;">75.2</td>
<td style="text-align: left;">72.7</td>
<td style="text-align: left;">65.3</td>
<td style="text-align: left;">76.2</td>
<td style="text-align: left;">76.5</td>
<td style="text-align: left;">–</td>
<td style="text-align: left;">75.4</td>
</tr>
<tr class="odd">
<td style="text-align: center;">GN <span class="citation" data-cites="GN"></span></td>
<td style="text-align: left;">75.9</td>
<td style="text-align: left;">75.8</td>
<td style="text-align: left;">76.0</td>
<td style="text-align: left;">75.8</td>
<td style="text-align: left;"><strong>75.9</strong></td>
<td style="text-align: left;">75.9</td>
<td style="text-align: left;">75.8</td>
<td style="text-align: left;"><strong>75.5</strong></td>
<td style="text-align: left;">75.5</td>
</tr>
<tr class="even">
<td style="text-align: center;">SN</td>
<td style="text-align: left;"><strong>76.9</strong></td>
<td style="text-align: left;"><strong>76.7</strong></td>
<td style="text-align: left;"><strong>76.7</strong></td>
<td style="text-align: left;"><strong>75.9</strong></td>
<td style="text-align: left;">75.6</td>
<td style="text-align: left;"><strong>76.3</strong></td>
<td style="text-align: left;"><strong>76.6</strong></td>
<td style="text-align: left;">75.0<sup>*</sup></td>
<td style="text-align: left;"><strong>75.9</strong></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span>GN</span><span class="math inline">−</span><span>BN</span></td>
<td style="text-align: left;">-0.5</td>
<td style="text-align: left;">-0.5</td>
<td style="text-align: left;">0.8</td>
<td style="text-align: left;">3.1</td>
<td style="text-align: left;"><span>10.6</span></td>
<td style="text-align: left;">-0.3</td>
<td style="text-align: left;">-0.7</td>
<td style="text-align: left;">–</td>
<td style="text-align: left;">0.1</td>
</tr>
<tr class="even">
<td style="text-align: left;"><span>SN</span><span class="math inline">−</span><span>BN</span></td>
<td style="text-align: left;"><span>0.5</span></td>
<td style="text-align: left;"><span>0.4</span></td>
<td style="text-align: left;"><span>1.5</span></td>
<td style="text-align: left;">3.2</td>
<td style="text-align: left;">10.3</td>
<td style="text-align: left;">0.1</td>
<td style="text-align: left;">0.1</td>
<td style="text-align: left;">–</td>
<td style="text-align: left;">0.5</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><span>SN</span><span class="math inline">−</span><span>GN</span></td>
<td style="text-align: left;"><span>1.0</span></td>
<td style="text-align: left;"><span>0.9</span></td>
<td style="text-align: left;"><span>0.7</span></td>
<td style="text-align: left;">0.1</td>
<td style="text-align: left;">-0.3</td>
<td style="text-align: left;">0.4</td>
<td style="text-align: left;">0.8</td>
<td style="text-align: left;">-0.5</td>
<td style="text-align: left;">0.4</td>
</tr>
</tbody>
</table>
*For (8,1), SN contains IN and SN without BN, as BN is the same as IN in training.


## Getting Started
* Install [PyTorch](http://pytorch.org/)
* Clone the repo:
  ```
  git clone https://github.com/switchablenorms/Switchable-Normalization.git
  ```

### Requirements
- python packages
  - pytorch>=0.4.0
  - torchvision>=0.2.1
  - tensorboardX
  
### Data Preparation
- Download the ImageNet dataset and put them into the `{repo_root}/data/imagenet`.
  - move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Model Zoo
| Model | Top-1<sup>*</sup> | Top-5<sup>*</sup> | Epochs |LR Scheduler| Weight Decay | Download | 
| :--:  | :--: | :--:  | :--:  | :--:  | :--:  | :--: |
|ResNet50v1+SN (8,32) | 76.92% | 93.26% | 100  | Initial lr=0.1 decay=0.1 steps[30,60,90,10]| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1lOTzjgX6B9J9gkm8JdxaWGBKC1T9VLsl)  [[Baidu Pan]](https://pan.baidu.com/s/1pLdnZYxynpztEnc1eUzVvA)|
|ResNet50v1+SN (8,32) | 77.49% | 93.32% | 120  | warmup + cosine lr| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=17mHmoVom2zM7nrbFeE4yzKa7KtqykTyD)  [[Baidu Pan]](https://pan.baidu.com/s/1jx3Bj15hgfEBZYhi5HP0kQ)|

*1-crop validation accuracy on ImageNet (center 224x224 crop from resized image with shorter side=256)     

When evaluation, download them and put them into the `{repo_root}/data/pretrained_model`.

## License

All materials in this repository are released under the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/).

