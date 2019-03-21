ALGO="resnet"$1$2
python3 -u gen_megaface.py --backbone=resnet$1 --norm-func=$2 --algo "$ALGO"  --checkpoint-path $3
python3 -u remove_noises.py --algo "$ALGO"
cd 'devkit/experiments/'
python -u run_experiment.py ../../"$ALGO"_feature_out_clean/megaface/ ../../"$ALGO"_feature_out_clean/facescrub/ _"$ALGO".bin ../../"$ALGO"_result -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -