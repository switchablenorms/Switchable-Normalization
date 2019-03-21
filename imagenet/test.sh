now=$(date +"%Y%m%d_%H%M%S")
python -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py  \
--config $1 -e 2>&1|tee test-$now.log




