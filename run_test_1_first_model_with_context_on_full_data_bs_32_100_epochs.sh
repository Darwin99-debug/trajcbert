#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:1
#SBATCH --time=4-0:00
#SBATCH --job-name=trajcbert_test_full_data_1_point_full_context_bs_16_1_epochs
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=16 # number of cores for each task
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca




module load scipy-stack/2023a
VENV_DIR=$SLURM_TMPDIR/MYENV
virtualenv  $VENV_DIR
source $VENV_DIR/bin/activate
pip install -r requirements.txt --no-index
pip list

PRETRAINED_MODEL_NAME="/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version_full_bs_32_20_epochs_with_context"
DATALOADER_DIR="/home/daril/scratch/data/trajcbert/test_dataloader/test_dataloader_full_32_bs.pt"


python testing_1_point_with_context_generic.py \
--pretrained_model_name $PRETRAINED_MODEL_NAME \
--dataloader_dir $DATALOADER_DIR


