#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -o fair_outs/PGEX_fair.out
#SBATCH -e fair_outs/PGEX_fair.err
#SBATCH -J PGEX_fair

source activate GXAI
python3 eval_SG_fairness.py --exp_method PGEX --model GIN --save_dir fairness_results
