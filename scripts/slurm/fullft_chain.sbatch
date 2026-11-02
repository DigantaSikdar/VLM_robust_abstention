#!/usr/bin/env bash
#SBATCH --job-name=fullft-chain
#SBATCH --account=advdls25
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6-00:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# Several full-FT runs back to back in ONE 2-GPU allocation, so every run uses
# the same two GPUs and none waits in the queue again. Each stage is one recipe
# (its own vLLM server + trainer via train_fullft_2gpu.sbatch). A stage whose
# final checkpoint already exists is skipped. A stage that fails within 30 min
# (a setup problem that would repeat) stops the chain; a later failure is logged
# and the chain moves on.
#   sbatch scripts/slurm/fullft_chain.sbatch recipe1.yaml recipe2.yaml ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "node: $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
for CFG in "$@"; do
  OUT=$(/gscratch/dsikdar/caad_env/bin/python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CFG")
  if [ -f "$OUT/checkpoints/final/model.safetensors.index.json" ] || [ -f "$OUT/checkpoints/final/model.safetensors" ]; then
    echo "== skip $CFG (final checkpoint exists)"; continue
  fi
  echo "== $(date '+%F %T') start $CFG"
  T0=$SECONDS
  bash scripts/slurm/train_fullft_2gpu.sbatch "$CFG"; RC=$?
  echo "== $(date '+%F %T') end $CFG  exit=$RC  ($(( (SECONDS - T0) / 60 )) min)"
  if [ "$RC" -ne 0 ] && [ $((SECONDS - T0)) -lt 1800 ]; then
    echo "== stage failed within 30 min (setup problem, would repeat): stopping the chain"; exit "$RC"
  fi
  sleep 15                                   # let the stage's vLLM server release GPU 0 and its port
done
