now=$(date +"%Y%m%d_%H%M%S")

if [ ! -d "log" ]; then
  mkdir log
fi

model_dir=log/resnet_v1_sn_50_moving_average/

python -u main.py \
-m resnet_v1_sn_50 -j 16 -b 256 --epochs 100 -md $model_dir --using-moving-average \
data/imagenet \
2>&1 | tee log/resnet_v1_sn_50_moving_average-$now.log
