#!/bin/bash
#SBATCH --job-name=kiln-dpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
# Plain torchrun for the offline DPO path (no Ray, no rollout engines).
set -euo pipefail
export MODEL_PATH=${MODEL_PATH:-/models/Qwen3-8B} DATA_DIR=${DATA_DIR:-/data}
torchrun --nproc_per_node=8 --standalone -m kiln.cli train configs/dpo_offline.yaml
