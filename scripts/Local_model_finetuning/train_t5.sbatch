#!/bin/bash

#SBATCH --account=users
#SBATCH --job-name=t5-finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=train_t5_%j.out
#SBATCH --error=train_t5_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

# Load modules
module purge

# Setup Singularity for GPU with PyTorch
SINGULARITY_IMAGE=/scratch/work/public/singularity/ubuntu-20.04.1.sif
OVERLAY_FILE=/scratch/work/public/examples/greene-getting-started/overlay-15GB-500K-pytorch.ext3

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Run training script inside Singularity container with GPU support
singularity exec --nv \
    --overlay $OVERLAY_FILE:ro \
    $SINGULARITY_IMAGE /bin/bash << 'EOF'

# Activate conda environment
source /ext3/miniconda3/bin/activate

# Install required packages if not already installed
pip install --quiet transformers datasets accelerate evaluate rouge-score sentencepiece openpyxl

# Run the training script
python scripts/Local_model_finetuning/train_t5.py \
    --epochs 5 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 3e-4 \
    --fp16 \
    --output-dir /scratch/$USER/t5-models

EOF

echo ""
echo "End time: $(date)"
echo "Job completed!"
