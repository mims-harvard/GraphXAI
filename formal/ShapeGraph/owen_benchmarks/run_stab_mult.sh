#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-5:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o GBP2.out
#SBATCH -e GBP2.err
#SBATCH -J GBP2

source activate GXAI
python3 stab_mult.py --exp_method GBP --model GIN --num_splits 3 --my_split 2
