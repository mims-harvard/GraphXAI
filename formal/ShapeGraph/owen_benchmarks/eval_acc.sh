#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_SUBX.out
#SBATCH -e eval_SUBX.err
#SBATCH -J SUBX_stab

source activate GXAI
python3 eval_SG_accuracy.py --exp_method SUBX --model GIN --save_dir SUBX/SUBX_results/acc
