#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o PGMEX.out
#SBATCH -e PGMEX.err
#SBATCH -J PGMEX_exps

source activate GXAI
python3 eval_SG_exps.py --exp_method PGMEX --model GIN
