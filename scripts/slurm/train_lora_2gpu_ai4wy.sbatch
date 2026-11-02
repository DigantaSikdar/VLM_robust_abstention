#!/usr/bin/env bash
#SBATCH --job-name=caad-lora
#SBATCH --account=uwyo-0010
#SBATCH --partition=gp-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# LoRA training on one node, two GPUs (same layout as train_fullft_2gpu_ai4wy.sbatch):
#   GPU 0  vLLM rollout server with --enable-lora (the student adapter is reloaded every rollout.sync_every
#          steps) + the frozen external teacher or GRPO KL reference, if the recipe has one (teacher.device: cuda:1)
#   GPU 1  student training (bf16 base, fp32 LoRA adapters)
# Inside the trainer CUDA_VISIBLE_DEVICES=1,0, so cuda:0 = training GPU, cuda:1 = server GPU.
#   sbatch scripts/slurm/train_lora_2gpu_ai4wy.sbatch <recipe.yaml> [dotted overrides...]
set -euo pipefail
CFG="${1:?usage: sbatch ... train_fullft_2gpu.sbatch <recipe.yaml> [overrides]}"; shift
TRAIN_PY="${TRAIN_PY:-/gscratch/dsikdar/caad_env/bin/python}"
VLLM_BIN="${VLLM_BIN:-/gscratch/dsikdar/vllm_env/bin/vllm}"
VLLM_MEM="${VLLM_MEM:-0.55}"                 # leaves ~35 GB of the server GPU for the teacher
PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${SLURM_JOB_ID:-$$}"   # per-job compile cache: the shared ~/.cache one races between concurrent jobs on NFS (stale handles)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DIR="${WANDB_DIR:-/gscratch/dsikdar/wandb}"; mkdir -p "$WANDB_DIR"
export WANDB_MODE="${WANDB_MODE:-offline}"   # no W&B login on this cluster; upload later with wandb sync
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "node: $(hostname)"
MODEL=$("$TRAIN_PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['name'])" "$CFG")
RANK=$("$TRAIN_PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['lora']['r'])" "$CFG")
SLOG="/gscratch/dsikdar/logs/vllm-server-${SLURM_JOB_ID:-local}-$(basename "$CFG" .yaml).log"

# media path: /gscratch is a link to /cluster/ai4wy/gscratch and caad_data holds links into the backup
echo ">> [1/2] vLLM server on GPU 0, port $PORT (log: $SLOG)"
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_BIN" serve "$MODEL" --port "$PORT" --dtype bfloat16 --seed 0 \
    --gpu-memory-utilization "$VLLM_MEM" --max-model-len 18432 \
    --enable-lora --max-lora-rank "$RANK" --max-loras 2 --max-cpu-loras 4 \
    --limit-mm-per-prompt '{"image": 1}' \
    --allowed-local-media-path /cluster/ai4wy/gscratch/dsikdar > "$SLOG" 2>&1 &
SERVER_PID=$!
trap 'echo ">> stopping vLLM server"; kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   server healthy after $((i*5))s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "!! vLLM server died:"; tail -30 "$SLOG"; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server not healthy"; tail -30 "$SLOG"; exit 1; }

echo ">> [2/2] training on GPU 1"
CUDA_VISIBLE_DEVICES=1,0 PYTHONPATH=src "$TRAIN_PY" -W ignore::FutureWarning -m caad.train \
  --config "$CFG" rollout.server.port=$PORT "$@"
