#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-06:00
#SBATCH -p short
#SBATCH --mem=10G
#SBATCH -o eval_PGEX.out
#SBATCH -e eval_PGEX.err
#SBATCH -J PGEX

source activate GXAI
python3 eval_SG_stability.py --exp_method pgex --model GIN --model_path /home/owq978/GraphXAI/formal/model_homophily.pth 
