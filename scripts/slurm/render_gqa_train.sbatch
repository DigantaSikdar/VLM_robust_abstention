#!/usr/bin/env bash
#SBATCH --job-name=gqa-train-render
#SBATCH --account=advdls25
#SBATCH --partition=teton
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/gscratch/dsikdar/logs/%x-%A_%a.out
#
# Render the CAAD GQA training set (clean + corrupted PNGs) from its manifest.
# Resumable. Run `python -m caad.data.prepare_gqa_train assign ...` first.
#   OUT=/gscratch/dsikdar/caad_data/gqa_train_30k sbatch scripts/slurm/render_gqa_train.sbatch
set -euo pipefail
PY="${PY:-/gscratch/dsikdar/caad_env/bin/python}"
OUT="${OUT:-/gscratch/dsikdar/caad_data/gqa_train_30k}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
PYTHONPATH=src "$PY" -W ignore -m caad.data.prepare_gqa_train render --out "$OUT" \
  --workers "$((SLURM_CPUS_PER_TASK - 2))" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
