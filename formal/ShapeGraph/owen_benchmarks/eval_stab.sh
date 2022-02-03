#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=10G
#SBATCH -o eval_RAND.out
#SBATCH -e eval_RAND.err
#SBATCH -J RAND

source activate GXAI
python3 eval_SG_stability.py --exp_method RAND --model GIN --model_path /home/owq978/GraphXAI/formal/model_homophily.pth 
