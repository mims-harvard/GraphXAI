#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o eval_GLIME.out
#SBATCH -e eval_GLIME.err
#SBATCH -J GLIME

source activate GXAI
python3 eval_SG_stability.py --exp_method GLIME --model GIN --model_path /home/owq978/GraphXAI/formal/model_homophily.pth 
