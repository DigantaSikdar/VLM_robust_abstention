#!/usr/bin/env bash
#SBATCH --job-name=steer-full
#SBATCH --account=uwyo-0010
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Full steering evaluation (README "Step 3") for one model and one shard, all three
# views and the five paper configs. Resumable: resubmit the same command to continue.
#   sbatch -J steer-frkl-s0 scripts/slurm/steer_full_ai4wy.sbatch <checkpoint dir> <tag> <shard> ["image clean noimg"]
# NORM_SCALE (default 1): extra steering factor for a model whose residual norm differs from Qwen3-VL (Gemma 3).
set -euo pipefail
M="${1:?checkpoint dir}"; TAG="${2:?tag}"; SHARD="${3:?shard}"; VIEWS="${4:-image clean noimg}"
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
for VIEW in $VIEWS; do
  /gscratch/dsikdar/caad_env/bin/python -W ignore -m caad.eval.steer_full --lora "$M" \
    --vectors "outputs/steer_full/$TAG" --norm-ref outputs/steer_full/rkl --norm-scale "${NORM_SCALE:-1}" --out "outputs/steer_full/$TAG/eval" \
    --view "$VIEW" --configs trained contrast:single:18:8:0.5 contrast:single:18:8:-1 noimg:single:18:1:0.5 noimg:single:18:1:-1 \
    --shard "$SHARD" --num-shards "${NUM_SHARDS:-8}" --batch 8
done
