# Switchable Normalization

Switchable Normalization is a normalization technique that is able to learn different normalization operations for different normalization layers in a deep neural network in an end-to-end manner.

![](teaser.png?raw=true)

## Update

- 2019/3/21: Release distributed training framework and face recognition framework. We also release a pytorch implementation of **SyncBN** and **SyncSN** for small batch tasks such as segmentation and detection. More details about **SyncBN** and **SyncSN** can refer to [this](http://htmlpreview.github.io/?https://github.com/JiaminRen/SyncSN/blob/master/syncBNsyncSN.html).
- 2018/7/27: The pretrained models of ResNet50+SN(8,1) and SN(8,4) have been released. These models may help in the finetuning stage when the batch size of a target task is constrained to be small. We also release the pretrained models of ResNet101v2+SN that achieves 78.81%/94.16% top-1/top-5 accuracies on ImageNet. More pretrained models will be released soon!
- 2018/7/26: The code for object detection have been released in the repository of [SwitchNorm_Detection](https://github.com/switchablenorms/SwitchNorm_Detection).
- 2018/7/9: We would like to explain the merit behind SN. See [html preview](http://htmlpreview.github.io/?https://github.com/switchablenorms/Switchable-Normalization/blob/master/blog_cn/blog_cn.html) or [this blog (in Chinese)](https://zhuanlan.zhihu.com/p/39296570?utm_source=wechat_session&utm_medium=social&utm_oi=70591319113728).
- 2018/7/4: Model zoo updated!
- 2018/7/2: The code of image classification and a pretrained model on ImageNet are released.

## Citation

This repository provides imagenet classification results and models trained with [Switchable Normalization](https://arxiv.org/abs/1806.10779). You are encouraged to cite the following paper if you use SN in research.

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng and Ruimao Zhang and Jingyu Li},
  journal={International Conference on Learning Representation (ICLR)},
  year={2019}
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
*For (8,1), SN contains IN and LN without BN, as BN is the same as IN in training.



## Model Zoo

We provide models pretrained with SN on ImageNet, and compare to those pretrained with BN as reference. If you use these models in research, please cite the SN paper. The configuration of SN is denoted as (#GPUs, #images per GPU).

| Model | Top-1<sup>*</sup> | Top-5<sup>*</sup> | Epochs |LR Scheduler| Weight Decay | Download |
| :----:  | :--: | :--:  | :--:  | :--:  | :--:  | :--: |
|ResNet101v2+SN (8,32) | 78.81% | 94.16% | 120  | warmup + cosine lr| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1z-F1nc7T-cdEM-HtYDALW0pJGPX0-RkS)  [[Baidu Pan]](https://pan.baidu.com/s/1rK-ukAjEIPql2ECi38hRbQ)|
|ResNet101v1+SN (8,32) | 78.54% | 94.10% | 120  | warmup + cosine lr| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1xza5ju6lRRelPym3Sj99VY_8c4jXhxkn)  [[Baidu Pan]](https://pan.baidu.com/s/1CCM0zZITQOzkXd__pWERZw)|
|ResNet50v2+SN (8,32) | 77.57% | 93.65% | 120  | warmup + cosine lr| 1e-4 |[[Google Drive]](https://drive.google.com/file/d/1CjLURwDkynhiXoKjYxUyEIOnWDPVmPnL/view?usp=sharing)  [[Baidu Pan]](https://pan.baidu.com/s/1XdN4tQcvTTIZjyNrnQpQXw)|
|ResNet50v1+SN (8,32) | 77.49% | 93.32% | 120  | warmup + cosine lr| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=17mHmoVom2zM7nrbFeE4yzKa7KtqykTyD)  [[Baidu Pan]](https://pan.baidu.com/s/1jx3Bj15hgfEBZYhi5HP0kQ)|
|ResNet50v1+SN (8,32) | 76.92% | 93.26% | 100  | Initial lr=0.1 decay=0.1 steps[30,60,90,10]| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1lOTzjgX6B9J9gkm8JdxaWGBKC1T9VLsl)  [[Baidu Pan]](https://pan.baidu.com/s/1pLdnZYxynpztEnc1eUzVvA)|
|ResNet50v1+SN (8,4) | 75.85% | 92.7% | 100  | Initial lr=0.0125 decay=0.1 steps[30,60,90,10]| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1x_ZR0ZFtvRzZHkFZVXeVoLeXoJKX-LrZ)  [[Baidu Pan]](https://pan.baidu.com/s/1qdzaB9r6DffTm_DXTG7MAw)|
|ResNet50v1+SN (8,1)<sup>†</sup> | 75.94% | 92.7% | 100  | Initial lr=0.003125 decay=0.1 steps[30,60,90,10]| 1e-4 |[[Google Drive]](https://drive.google.com/open?id=1-y8dDn19Hx02efy4Z5ZRihggYHTYvqkc)  [[Baidu Pan]](https://pan.baidu.com/s/10JmkrQ2WWEeWzqNfB0Fytg)|
|ResNet50v1+BN | 75.20% | 92.20% | --  | stepwise decay | -- |[[TensorFlow models]](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|
|ResNet50v1+BN | 76.00% | 92.98% | --  | stepwise decay | -- |[[PyTorch Vision]](https://github.com/Cadene/pretrained-models.pytorch#torchvision)|
|ResNet50v1+BN | 75.30% | 92.20% | --  | stepwise decay | -- |[[MSRA]](https://github.com/KaimingHe/deep-residual-networks)|
|ResNet50v1+BN | 75.99% | 92.98% | --  | stepwise decay | -- |[[FB Torch]](https://github.com/facebook/fb.resnet.torch)|

*single-crop validation accuracy on ImageNet (a 224x224 center crop from resized image with shorter side=256)

†For (8,1), SN contains IN and LN without BN, as BN is the same as IN in training. When using this model, you should add `using_bn : False` in  yaml file.

## License

All materials in this repository are released under the [CC-BY-NC 4.0 LICENSE](https://creativecommons.org/licenses/by-nc/4.0/).

