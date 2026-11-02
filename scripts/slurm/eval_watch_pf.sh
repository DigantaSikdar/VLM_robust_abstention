#!/usr/bin/env bash
# Same as eval_watch.sh, with the prompt format taken from $PF (default think_answer).
# Polls for FINAL full-FT checkpoints and submits the standard evaluation
# (main / vqav2+vizwiz / no-image, system prompt) for each as soon as it appears.
#   sbatch -A advdls25 -p <cpu partition> -t 4-00:00:00 scripts/slurm/eval_watch.sh tag1=/run/dir1 tag2=/run/dir2 ...
set -uo pipefail
PF="${PF:-think_answer}"
cd "${SLURM_SUBMIT_DIR:-.}"
A="-A advdls25 -p beartooth-gpu,mb-a30,wildiris --exclude=mba30-005 --gres=gpu:a30:1 -c 8 --mem=64G"
while :; do
  left=0
  for pair in "$@"; do
    T=${pair%%=*}; D=${pair#*=}; M=$D/checkpoints/final; MARK=$D/.eval_submitted
    [ -f "$MARK" ] && continue
    if [ -f "$M/model.safetensors.index.json" ] && [ -f "$M/preprocessor_config.json" ]; then
      sleep 120                                  # let the save finish flushing
      a=$(sbatch --parsable $A -t 08:00:00 -J eval-$T --export=ALL,MODEL=$M,OUT=outputs/eval/$T-student scripts/slurm/eval_vqa_c.sbatch --prompt-format "$PF")
      b=$(sbatch --parsable $A -t 04:00:00 -J extra-$T --export=ALL,MODEL=$M,OUT=outputs/eval/$T-extra scripts/slurm/eval_vqa_c.sbatch --prompt-format "$PF" --datasets vqav2,vizwiz)
      c=$(sbatch --parsable $A -t 04:00:00 -J noimg-$T --export=ALL,MODEL=$M,OUT=outputs/eval/noimg-$T scripts/slurm/eval_vqa_c.sbatch --prompt-format "$PF" --no-image --datasets textvqa,gqa,vqav2,vizwiz)
      echo "$(date '+%F %T') $T -> $a $b $c" | tee "$MARK"
    else
      left=$((left + 1))
    fi
  done
  [ "$left" -eq 0 ] && { echo "all submitted"; exit 0; }
  sleep 600
done
