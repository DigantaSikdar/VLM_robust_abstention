#!/usr/bin/env bash
#SBATCH --job-name=cs-generate
#SBATCH --account=advdls25
#SBATCH --partition=beartooth-gpu
#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Teacher rollouts on clean images for the SFT cold start, then filter. Resumable.
#   OUT=/gscratch/dsikdar/caad_data/coldstart_gqa30k sbatch scripts/slurm/coldstart_generate.sbatch
set -euo pipefail
OUT="${OUT:-/gscratch/dsikdar/caad_data/coldstart_gqa30k}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PF="${PROMPT_FORMAT:-inline}"
export HF_HOME=/gscratch/dsikdar/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# one GPU when it fits the model (>40 GB); tensor parallel only on A30 pairs. Multi-GPU
# vLLM on the H100 nodes JIT-compiles communication kernels and needs nvcc, which they lack.
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NG=$(nvidia-smi -L | wc -l)
if [ "$MEM" -gt 40000 ]; then TP=1; export CUDA_VISIBLE_DEVICES=$(echo ${CUDA_VISIBLE_DEVICES:-0} | cut -d, -f1); else TP=$NG; fi
echo "tensor parallel: $TP"
/gscratch/dsikdar/vllm_env/bin/python -m caad.data.coldstart generate --out "$OUT" --model "$MODEL" --prompt-format "$PF" --tp "$TP"
/gscratch/dsikdar/vllm_env/bin/python -m caad.data.coldstart filter --out "$OUT" --model "$MODEL" --prompt-format "$PF"
