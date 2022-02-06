#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o fair_outs/GNNEX3_group.out
#SBATCH -e fair_outs/GNNEX3_group.err
#SBATCH -J GNNEX3_group

source activate GXAI
python3 fair_mult.py --exp_method GNNEX --model GIN --save_dir fairness_results --num_splits 4 --my_split 3 --ignore_cf
