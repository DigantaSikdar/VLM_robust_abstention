#!/usr/bin/env bash
#SBATCH --job-name=sft
#SBATCH --account=advdls25
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Supervised cold start: sbatch scripts/slurm/sft.sbatch configs/sft/coldstart_8b_qwen3vl4b.yaml [key=value ...]
set -euo pipefail
CFG="$1"; shift
export HF_HOME=/gscratch/dsikdar/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/gscratch/dsikdar/caad_env/bin/python -m caad.sft --config "$CFG" "$@"
