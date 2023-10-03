#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --job-name=generating_test_data_for_1_2_bz_32_20_epochs
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
# pip list



DIR_INPUTS_IDS='/home/daril/trajcbert/savings_for_parallel_computing_full/input_ids_full_opti.pkl'
DIR_ATTENTION_MASKS='/home/daril/trajcbert/savings_for_parallel_computing_full/attention_masks_full_opti.pkl'
DIR_TARGETS='/home/daril/trajcbert/savings_for_parallel_computing_full/targets_full_opti.pkl'
DATALOADER_DIR='/home/daril/scratch/data/trajcbert/test_dataloader/test_dataloader_full_32_bs.pt'


python genrate_test_data_for_initial_model.py \
--inputs_ids_path $DIR_INPUTS_IDS \
--attention_masks_path $DIR_ATTENTION_MASKS \
--targets_path $DIR_TARGETS \
--dataloader_dir $DATALOADER_DIR



