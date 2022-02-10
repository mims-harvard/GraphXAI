#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_CAM.out
#SBATCH -e eval_CAM.err
#SBATCH -J CAM_stab

source activate GXAI
python3 eval_SG_stability.py --exp_method CAM --model GIN
