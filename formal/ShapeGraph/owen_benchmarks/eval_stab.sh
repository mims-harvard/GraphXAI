#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_PGMEX.out
#SBATCH -e eval_PGMEX.err
#SBATCH -J PGMEX

source activate GXAI
python3 eval_SG_stability.py --exp_method PGMEX --model GIN
