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
conda activate sglang
pwd

python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B --port 30000 --host 0.0.0.0 &
SERVER_PID=$!

echo "STARTED SERVER with PID $SERVER_PID"

wait_for_server() {
    while true; do
        if curl -s -o /dev/null -w "%{http_code}" http://0.0.0.0:30000/health >/dev/null 2>&1; then
            echo "server ready"
            break
        else
            echo "waiting for server"
            sleep 1
        fi
    done
}

wait_for_server
cd /iris/u/rypark/code/hover-rl/distill
source ../env/bin/activate
python3 generate.py
