#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX_cf0.out
#SBATCH -e SUBX_cf0.err
#SBATCH -J SUBX_cf0

source activate GXAI
python3 fair_mult_SG.py --save_dir SUBX_results --num_splits 4 --my_split 0 --ignore_group
