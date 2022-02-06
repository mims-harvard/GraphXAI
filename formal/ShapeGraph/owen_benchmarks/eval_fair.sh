#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o fair_PGEX.out
#SBATCH -e fair_PGEX.err
#SBATCH -J PGEX_fair

source activate GXAI
python3 eval_SG_fairness.py --exp_method PGEX --model GIN --save_dir fairness_results