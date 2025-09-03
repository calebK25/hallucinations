#!/bin/bash
#SBATCH --job-name=DRM_EXPERIMENT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ck2867@princeton.edu

# Module environment
module purge
module load anaconda3/2024.6
module load cudatoolkit/11.8
conda activate hallucination

# Hugging Face cache
export HF_HOME=/scratch/gpfs/$USER/models2/hub
export HF_TOKEN=${HF_TOKEN}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "$HF_HOME"

# Diagnostics
echo "Job started : $(date)"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "HF_HOME     : $HF_HOME"

# Command
python hallucination_test_models_full_experiment.py

echo "Job finished : $(date)" 