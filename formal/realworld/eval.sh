#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX_benzene.out
#SBATCH -e SUBX_benzene.err
#SBATCH -J SUBX_benzene

source activate GXAI
python3 graph_eval.py --exp_method SUBX --model GIN --dataset benzene --accuracy --faithfulness
