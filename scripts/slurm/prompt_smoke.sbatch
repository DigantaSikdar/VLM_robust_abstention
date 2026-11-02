#!/usr/bin/env bash
#SBATCH --job-name=prompt-smoke
#SBATCH --account=advdls25
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Prompt-format smoke test: same first N questions per dataset, one run per format.
#   MODEL=Qwen/Qwen3-VL-8B-Instruct TAG=8b sbatch -p mb-l40s --gres=gpu:1 scripts/slurm/prompt_smoke.sbatch
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"; TAG="${TAG:-4b}"; N="${N:-300}"
FORMATS="${FORMATS:-inline_brief think_answer}"
export HF_HOME=/gscratch/dsikdar/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=src
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TP=$(nvidia-smi -L | wc -l)
for f in $FORMATS; do
  /gscratch/dsikdar/vllm_env/bin/python -W ignore -m caad.eval.vllm_eval run --root /gscratch/dsikdar/caad_data/vqa_c \
    --model "$MODEL" --out "outputs/prompt_smoke/${TAG}_${f}" --prompt-format "$f" \
    --datasets textvqa,gqa,vqav2 --limit "$N" --tp "$TP"
done
