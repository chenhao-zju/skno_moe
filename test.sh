config_file=./config/SKNO.yaml
config='afno_backbone'
run_num='1'

NAME='skno_moe_93to09'

LOG_DIR="/home/bingxing2/ailab/group/ai4earth/haochen/logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

WEIGHTS="/home/bingxing2/ailab/group/ai4earth/haochen/logs/skno_moe_93to09/afno_backbone/1/training_checkpoints/best_ckpt.tar"

CUDA_VISIBLE_DEVICES=0 nohup /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python test.py \
            --yaml_config=$config_file --config=$config --run_num=$run_num --override_dir=$LOG_DIR \
            --weights=$WEIGHTS > ${LOG_DIR}test.log 2>&1 &