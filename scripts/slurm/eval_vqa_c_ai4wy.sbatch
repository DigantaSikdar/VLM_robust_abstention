#!/usr/bin/env bash
#SBATCH --job-name=vqa-c-eval
#SBATCH --account=uwyo-0010
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Full TextVQA-C / GQA-C evaluation with vLLM on one GPU (35,156 generations).
# Runs entirely on the cluster; resumable (resubmit to continue).
#   MODEL=Qwen/Qwen3-VL-4B-Instruct sbatch scripts/slurm/eval_vqa_c.sbatch
#   sbatch --dependency=afterok:<render_job> scripts/slurm/eval_vqa_c.sbatch
# Progress:  grep '\[progress\]' /gscratch/dsikdar/logs/vqa-c-eval-<jobid>.out | tail -1
# Results:   outputs/eval/<model>/summary.md
set -euo pipefail
PY="${PY:-/gscratch/dsikdar/vllm_env/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
ROOT="${ROOT:-/gscratch/dsikdar/caad_data/vqa_c}"
OUT="${OUT:-outputs/eval/$(basename "$MODEL")}"
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"   # per-job compile cache: the shared ~/.cache one races between concurrent jobs on NFS (stale handles)
# FlashInfer's sampler JIT-compiles CUDA kernels and needs nvcc, which the GPU nodes
# lack; greedy decoding doesn't need it, so use vLLM's built-in sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# fail fast if any corrupted image is missing
missing=$(python3 - "$ROOT" <<'PY'
import json, os, sys
n = 0
for ds in ("textvqa", "gqa"):
    for l in open(f"{sys.argv[1]}/{ds}/manifest.jsonl"):
        r = json.loads(l); n += not os.path.exists(r["corrupted_image"])
print(n)
PY
)
[ "$missing" = "0" ] || { echo "!! $missing corrupted images missing — run the render job first"; exit 1; }
PYTHONPATH=src "$PY" -W ignore -m caad.eval.vllm_eval run --root "$ROOT" --model "$MODEL" --out "$OUT" "$@"
