#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o SUBX_outs/SUBX1.out
#SBATCH -e SUBX_outs/SUBX1.err
#SBATCH -J SUBX1_stab

source activate GXAI
python3 stab_mult.py --exp_method SUBX --model GIN --save_dir SUBX_stab_results --num_splits 30 --my_split 0
