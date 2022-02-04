#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_IG.out
#SBATCH -e eval_IG.err
#SBATCH -J IG

source activate GXAI
python3 eval_SG_stability.py --exp_method IG --model GIN --model_path /home/owq978/GraphXAI/formal/model_homophily.pth 
