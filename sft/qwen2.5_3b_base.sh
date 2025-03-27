#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=240:00:00
#SBATCH --job-name=sft
#SBATCH --output=/iris/u/rypark/code/hover-rl/slurm/%j.out

EXP_NAME=qwen2.5_3b_base__sft__4o_mini_traces
echo $EXP_NAME

cd /iris/u/rypark/code/hover-rl/sft
source ~/.bashrc
conda activate verlc
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/iris/u/rypark/code/hover-rl/data/gpt_4o_mini_2024_07_18__train__filtered.parquet \
    data.val_files=/iris/u/rypark/code/hover-rl/data/gpt_4o_mini_2024_07_18__dev__filtered__lite.parquet \
    data.prompt_key=input \
    data.response_key=output \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=2048 \
    model.partial_pretrain=Qwen/Qwen2.5-3B \
    trainer.default_local_dir=/iris/u/rypark/code/hover-rl/models/$EXP_NAME \
    trainer.project_name=hover-rl \
    trainer.experiment_name=$EXP_NAME \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
