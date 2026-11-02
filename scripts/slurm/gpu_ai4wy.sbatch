#!/usr/bin/env bash
#SBATCH --job-name=caad-gpu
#SBATCH --account=uwyo-0010
#SBATCH --partition=gp-1,gp-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:00:00
#SBATCH --output=/gscratch/dsikdar/logs/%x-%j.out
#
# One-GPU wrapper for the analysis modules on the ai4wy cluster (GH200 nodes).
# Runs any caad module with the caad_env Python from the repo root, e.g.
#   sbatch -J steer-frkl scripts/slurm/gpu_ai4wy.sbatch caad.eval.steer_full --lora <ckpt> ...
set -euo pipefail
export HF_HOME="${HF_HOME:-/gscratch/dsikdar/hf}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/gscratch/dsikdar/caad_env/bin/python -W ignore -m "$@"
