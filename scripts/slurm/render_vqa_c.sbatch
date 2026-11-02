#!/usr/bin/env bash
#SBATCH --job-name=vqa-c-render
#SBATCH --account=advdls25
#SBATCH --partition=mb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Render every corrupted image of TextVQA-C / GQA-C (17,578 PNGs) from the
# fixed assignment in the manifests. Resumable: finished PNGs are skipped.
# Run `python -m caad.data.prepare_vqa_c assign ...` first.
#   sbatch scripts/slurm/render_vqa_c.sbatch
#   sbatch --array=0-3 scripts/slurm/render_vqa_c.sbatch    # 4 nodes
set -euo pipefail
PY="${PY:-/gscratch/dsikdar/caad_env/bin/python}"
OUT="${OUT:-/gscratch/dsikdar/caad_data/vqa_c}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 1 thread per worker
cd "$SLURM_SUBMIT_DIR"
PYTHONPATH=src "$PY" -W ignore -m caad.data.prepare_vqa_c render --out "$OUT" --datasets "${DATASETS:-textvqa,gqa}" \
  --workers "$((SLURM_CPUS_PER_TASK - 2))" --shard "$SHARD" --num-shards "$NSHARDS"
