#!/bin/bash
### module load anaconda/2021.11
### module load compilers/cuda/11.3
### module load cudnn/8.4.0.27_cuda11.x
### source activate yolov5
### 启用 IB 通信
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml

export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
### 获取每个节点的 hostname
for i in `scontrol show hostnames`
do
    let k=k+1
    host[$k]=$i
    echo ${host[$k]}
done

config_file=./config/SKNO.yaml
config='afno_backbone'
run_num='1'

NAME='skno_00to20_ddp_v2'

LOG_DIR="/home/bingxing2/ailab/group/ai4earth/haochen/logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

module unload compilers/gcc
source /home/bingxing2/apps/package/pytorch/1.13.1+cu117_cp38/env.sh
module unload compilers/gcc
source export_DDP_vars.sh

###主节点运行
/home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python -m torch.distributed.launch \
        --nnodes=2 --node_rank=0 --nproc_per_node=4 --master_addr="${host[1]}" --master_port="29501" \
        train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR \
        > ${LOG_DIR}train_1.log 2>&1 &


### 使用 srun 运行第二个节点
srun -N 1 --gres=gpu:4 -w ${host[2]} \
        /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python -m torch.distributed.launch \
        --nnodes=2 --node_rank=1 --nproc_per_node=4 --master_addr="${host[1]}" --master_port="29501" \
        train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR \
        > ${LOG_DIR}train_2.log 2>&1 &

wait
