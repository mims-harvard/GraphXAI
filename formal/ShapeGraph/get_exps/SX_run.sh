#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX.out
#SBATCH -e SUBX.err
#SBATCH -J SUBX_exps

source activate GXAI
python3 SX_exps.py --start_ind 0
