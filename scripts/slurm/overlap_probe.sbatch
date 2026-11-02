#!/usr/bin/env bash
#SBATCH --job-name=overlap
#SBATCH --account=advdls25
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Overlap-ratio probe (Li et al. 2026 metrics); all args go to caad.eval.overlap_probe.
#   sbatch scripts/slurm/overlap_probe.sbatch --out outputs/overlap/self_s0
#   sbatch --gres=gpu:a30:2 scripts/slurm/overlap_probe.sbatch --teacher-model Qwen/Qwen3-VL-8B-Instruct \
#       --teacher-device cuda:1 --out outputs/overlap/t8b_s0
set -euo pipefail
export HF_HOME=/gscratch/dsikdar/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/gscratch/dsikdar/caad_env/bin/python -m caad.eval.overlap_probe "$@"
