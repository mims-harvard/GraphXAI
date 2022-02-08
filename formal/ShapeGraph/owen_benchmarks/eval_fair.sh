#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o fair_outs/CAM_fair.out
#SBATCH -e fair_outs/CAM_fair.err
#SBATCH -J CAM_fair

source activate GXAI
python3 eval_SG_fairness.py --exp_method CAM --model GIN --save_dir fairness_results
