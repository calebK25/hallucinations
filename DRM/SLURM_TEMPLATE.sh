#!/bin/bash
# -----------------------------------------------------------------------------
# Generic SLURM template for DRM memory experiments
# -----------------------------------------------------------------------------
# Replace CAPITALISED placeholders as needed before submitting.
#
#  sbatch SLURM_TEMPLATE.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=DRM_EXPERIMENT          # descriptive name for the job
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # total tasks (1 per node is typical)
#SBATCH --cpus-per-task=4                  # CPU threads (tokenisation, etc.)
#SBATCH --mem=32G                          # system RAM per node
#SBATCH --gres=gpu:1                       # GPUs per node (adjust if multi-GPU)
#SBATCH --time=04:00:00                    # wall-time HH:MM:SS
#SBATCH --output=%x_%j.out                 # stdout file (%x=jobname %j=jobid)
#SBATCH --mail-type=FAIL,END               # notifications (optional)
#SBATCH --mail-user=YOU@EXAMPLE.COM        # your email

# ──────────────────────────────────────────────────────────────
#  Module environment / Conda activation
# ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.6
module load cudatoolkit/11.8

conda activate hallucination   # <<< make sure this env contains torch & transformers

# ──────────────────────────────────────────────────────────────
#  Hugging Face cache (shared scratch)
# ──────────────────────────────────────────────────────────────
export HF_HOME=/scratch/gpfs/$USER/models2/hub   # adjust path if needed
export HF_TOKEN=hf_xxx...                        # (only if you need gated models)
export TRANSFORMERS_OFFLINE=1                    # compute nodes have no internet
export HF_DATASETS_OFFLINE=1

# Ensure the directory exists
mkdir -p "$HF_HOME"

# ──────────────────────────────────────────────────────────────
#  Diagnostics (helpful for debugging)
# ──────────────────────────────────────────────────────────────
echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "HF_HOME     : $HF_HOME"

# ──────────────────────────────────────────────────────────────
#  Command – pick ONE of the following
# ──────────────────────────────────────────────────────────────
# 1) Full multi-model 2×2 experiment (long runtime)
# python hallucination_test_models_full_experiment.py

# 2) Single-model (OpenChat) experiment (short runtime)
python run_openchat_experiment.py

# ──────────────────────────────────────────────────────────────

echo "Job finished : $(date)" 