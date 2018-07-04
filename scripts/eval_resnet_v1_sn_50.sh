now=$(date +"%Y%m%d_%H%M%S")

python -u main.py \
-m resnet_v1_sn_50 -j 4 -b 256  -ckpt data/pretrained_model/ResNet50v1+SN\(8,32\)-77.49.pth  --evaluate \
data/imagenet \
2>&1 | tee resnet50_sn_eval-$now.log
