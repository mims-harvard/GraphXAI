#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_GNNEX.out
#SBATCH -e eval_GNNEX.err
#SBATCH -J GNNEX

source activate GXAI
python3 eval_SG_stability.py --exp_method GNNEX --model GIN
