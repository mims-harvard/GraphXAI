#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o mult_outs/SUBX_f49.out
#SBATCH -e mult_outs/SUBX_f49.err
#SBATCH -J SUBX_f49

source activate GXAI
python3 stability_saver.py --exp_method SUBX --model GIN --save_dir SUBX_results/stab --num_splits 50 --my_split 49
