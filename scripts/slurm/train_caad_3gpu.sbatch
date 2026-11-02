#!/usr/bin/env bash
#SBATCH --job-name=caad-train
#SBATCH --account=advdls25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# CAAD training on one node, THREE GPUs (for GPUs too small to hold student + 8B teacher):
#   GPU 0  vLLM rollout server (hot-loads the student LoRA adapter)
#   GPU 1  training (student forward + backward)
#   GPU 2  frozen external teacher (recipe: teacher.device: cuda:1; trainer sees CUDA_VISIBLE_DEVICES=1,2)
#   sbatch -p <partition> scripts/slurm/train_caad_3gpu.sbatch <recipe.yaml> [dotted overrides...]
set -euo pipefail
CFG="${1:?usage: sbatch ... train_caad_3gpu.sbatch <recipe.yaml> [overrides]}"; shift
TRAIN_PY="${TRAIN_PY:-/gscratch/dsikdar/caad_env/bin/python}"
VLLM_BIN="${VLLM_BIN:-/gscratch/dsikdar/vllm_env/bin/vllm}"
PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"   # per-job compile cache (shared ~/.cache races on NFS)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation; no change to the math
export WANDB_DIR="${WANDB_DIR:-/gscratch/dsikdar/wandb}"; mkdir -p "$WANDB_DIR"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
MODEL=$("$TRAIN_PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['name'])" "$CFG")
RANK=$("$TRAIN_PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['lora']['r'])" "$CFG")
SLOG="/gscratch/dsikdar/logs/vllm-server-${SLURM_JOB_ID:-local}.log"

echo ">> [1/2] vLLM server on GPU 0, port $PORT (log: $SLOG)"
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_BIN" serve "$MODEL" --port "$PORT" --dtype bfloat16 --seed 0 \
    --gpu-memory-utilization 0.85 --max-model-len 18432 \
    --enable-lora --max-lora-rank "$RANK" --max-loras 2 --max-cpu-loras 4 \
    --limit-mm-per-prompt '{"image": 1}' \
    --allowed-local-media-path /gscratch/dsikdar/caad_data > "$SLOG" 2>&1 &
SERVER_PID=$!
trap 'echo ">> stopping vLLM server"; kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   server healthy after $((i*5))s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "!! vLLM server died:"; tail -30 "$SLOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server not healthy"; tail -30 "$SLOG"; exit 1; }

echo ">> [2/2] training on GPU 1"
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH=src "$TRAIN_PY" -W ignore::FutureWarning -m caad.train \
  --config "$CFG" rollout.server.port=$PORT "$@"
