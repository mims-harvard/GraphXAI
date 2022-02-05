#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-16:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_PGEX.out
#SBATCH -e eval_PGEX.err
#SBATCH -J PGEX

source activate GXAI
python3 stab_mult.py --exp_method PGEX --model GIN --num_splits 3 --my_split 0