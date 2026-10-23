#!/bin/bash
#SBATCH --job-name=kiln-grpo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
# Launch a Ray cluster across the allocation, then run `kiln train --ray` on the head.
set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B}
export DATA_DIR=${DATA_DIR:-/data} CKPT_DIR=${CKPT_DIR:-/checkpoints}
export NCCL_IB_HCA=mlx5 NCCL_NET_GDR_LEVEL=5 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
export VLLM_USE_V1=1 OMP_NUM_THREADS=8

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head" hostname --ip-address | awk '{print $1}')
port=6379

echo "head: $head ($head_ip)"
srun --nodes=1 --ntasks=1 -w "$head" ray start --head --node-ip-address="$head_ip" --port=$port \
     --num-gpus=8 --block &
sleep 15
for ((i = 1; i < ${#nodes[@]}; i++)); do
  srun --nodes=1 --ntasks=1 -w "${nodes[$i]}" ray start --address="$head_ip:$port" --num-gpus=8 --block &
done
sleep 15

# Reward server on the head node's CPUs (rule verifiers only).
srun --nodes=1 --ntasks=1 -w "$head" --overlap kiln reward-serve configs/grpo_math_qwen3_8b.yaml --port 8008 &
export REWARD_URL="http://$head_ip:8008"
sleep 5

srun --nodes=1 --ntasks=1 -w "$head" --overlap \
  kiln train configs/grpo_math_qwen3_8b.yaml --ray --set logging.run_name="grpo-$SLURM_JOB_ID"
