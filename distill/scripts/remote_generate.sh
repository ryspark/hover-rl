#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=240:00:00
#SBATCH --job-name=sft
#SBATCH --output=/iris/u/rypark/code/hover-rl/slurm/%j.out

ulimit -n 64000
source ~/.bashrc
cd /iris/u/rypark/code/hover-rl/distill
source ../env/bin/activate
python3 generate.py
