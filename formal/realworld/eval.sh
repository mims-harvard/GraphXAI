#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-24:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX_fc.out
#SBATCH -e SUBX_fc.err
#SBATCH -J SUBX_fc

source activate GXAI
python3 graph_eval.py --exp_method SUBX --model GIN --dataset fc --accuracy --faithfulness
