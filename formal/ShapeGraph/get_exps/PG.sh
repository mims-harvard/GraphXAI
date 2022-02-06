#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o PGEX.out
#SBATCH -e PGEX.err
#SBATCH -J PGEX_exps

source activate GXAI
python3 PG_exps.py
