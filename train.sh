config_file=./config/SKNO.yaml
config='afno_backbone'
run_num='1'

NAME='skno_moe_93to09'

LOG_DIR="/home/bingxing2/ailab/group/ai4earth/haochen/logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py \
            --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR \
            > ${LOG_DIR}train_0721.log 2>&1 &

# salloc -N 1 --gres=gpu:4 -p vip_gpu_ailab -A ai4earth --qos=gpugpu \
#             nohup /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python train.py \
#             --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR \
#             > ${LOG_DIR}train_0716.log 2>&1 &