#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-03:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o ggf_outs/SUBX_fNUM.out
#SBATCH -e ggf_outs/SUBX_fNUM.err
#SBATCH -J SUBX_gNUM

source activate GXAI
python3 GGF_saver.py --exp_method SUBX --model GIN --save_dir SUBX_results/ggf --num_splits 50 --my_split NUM
