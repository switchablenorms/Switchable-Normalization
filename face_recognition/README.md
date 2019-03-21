# Switchable Normalization In Face Recognition
We use arcface* to evaluate SN in face recognition task. We used pytorch to reproduce the [insightface](https://github.com/deepinsight/insightface).
* Arcface: Additive angular margin loss for deep face recognition
J Deng, J Guo, N Xue, S Zafeiriou - arXiv preprint arXiv:1801.07698, 2018
## Training a model from scratch
```
./train.sh configs/config_resnet50bn.yaml
```
## Evaluating performance of a model
Download the pretrained models from Model Zoo and put them into the {repo_root}/face_recognition/data/pretrained_model
```
./test.sh configs/config_resnet50bn.yaml
```

## Model Zoo

We provide models pretrained with SN  on MS1M-ArcFace, and compare to those pretrained with BN as reference. If you use these models in research, please cite the SN paper. The configuration of SN is denoted as (#GPUs, #images per GPU).

| Model | MegaFace(%) | Epochs |LR Scheduler| Weight Decay | Download |
| :----:  | :--: | :--:  | :--:  | :--:  | :--:  |
|ResNet100+SYNCSN (16,32) | 98.51% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1mEREjkSjQDAcOUGgcz-LebQN9OGYweyX)  [[Baidu Pan]](https://pan.baidu.com/s/1jqd6QgKZ-UXbCiTfARdBYg)|
|ResNet100+SN (10,52) | 98.10% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1iulvp76q6Llun4nLxAWhO_GkxmbnuH0S)  [[Baidu Pan]](https://pan.baidu.com/s/1JELPufy8PB3hTYQHUiBUVg)|
|ResNet100+BN (8,64) | 98.29% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1zP_rZKWJRI155BFkn7j_pthlCoFMhBKU)  [[Baidu Pan]](https://pan.baidu.com/s/1BnlCa1DpFhH9VeJI5SA7bg)|
|ResNet50+SYNCSN (16,32) | 97.94% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://pan.baidu.com/s/1vYdyRg2pBZYz_71_Ozdt7Q)  [[Baidu Pan]](https://pan.baidu.com/s/1rK-ukAjEIPql2ECi38hRbQ)|
|ResNet50+SYNCSN+ARGMAX (8,64) | 98.26% | 10  | Initial lr=0.001 decay=0.1 steps[5,]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1JRRICZDHGRDx0JJzqY_zV0mzGFIGXfJY)  [[Baidu Pan]](https://pan.baidu.com/s/1wIpTdvmSbaB90akGbGehHw)|
|ResNet50+SN (8,64) | 97.84% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1fjxItr3ssLs2gNRQLwsWpgec2JQ_JFEM)  [[Baidu Pan]](https://pan.baidu.com/s/1ZngNCn_aH5NaRdFrPnH09Q)|
|ResNet50+BN (8,64) | 97.59% | 20  | Initial lr=0.1 decay=0.1 steps[12,15,18]| 5e-4 |[[Google Drive]](https://drive.google.com/open?id=1KjBSieShhBmW_UAnZYPIADtrI8OfvCg6)  [[Baidu Pan]](https://pan.baidu.com/s/1QMVucd6B4AV_VW9VOjohmA)|
