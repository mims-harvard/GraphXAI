#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-08:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o gcf_outs/SUBX_cNUM.out
#SBATCH -e gcf_outs/SUBX_cNUM.err
#SBATCH -J SUBX_cNUM

source activate GXAI
python3 gcf_saver.py --exp_method SUBX --model GIN --save_dir SUBX_results/gcf --num_splits 10 --my_split NUM
