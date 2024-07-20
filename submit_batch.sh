#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m4134_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J afno
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o afno_backbone_finetune.out

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='1'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
