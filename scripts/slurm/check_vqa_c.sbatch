#!/usr/bin/env bash
#SBATCH --job-name=vqa-c-check
#SBATCH --account=advdls25
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Check run: a few TextVQA-C / GQA-C questions, clean + corrupted, with the
# model's reasoning and the extracted answer saved for inspection.
#   MODEL=Qwen/Qwen3-VL-4B-Instruct N=12 sbatch scripts/slurm/check_vqa_c.sbatch
#   FORMAT=think_answer N=100 sbatch scripts/slurm/check_vqa_c.sbatch   # system-prompt format
set -euo pipefail
PY="${PY:-/gscratch/dsikdar/caad_env/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
N="${N:-12}"
FORMAT="${FORMAT:-inline}"
TAG="$(basename "$MODEL")-${FORMAT}"
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" OMP_NUM_THREADS=4
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=src "$PY" -W ignore -m caad.eval.sample_run \
  --root /gscratch/dsikdar/caad_data/vqa_c --model "$MODEL" --n "$N" --prompt-format "$FORMAT" \
  --out "outputs/check/${TAG}.jsonl"
