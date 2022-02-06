#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX-3.out
#SBATCH -e SUBX-3.err
#SBATCH -J SUBX-3

source activate GXAI
python3 SX.py --start_ind -3
