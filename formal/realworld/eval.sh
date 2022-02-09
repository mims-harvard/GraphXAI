#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX_mutag.out
#SBATCH -e SUBX_mutag.err
#SBATCH -J SUBX_mutag

source activate GXAI
python3 graph_eval.py --exp_method SUBX --model GIN --dataset mutag --accuracy --faithfulness
