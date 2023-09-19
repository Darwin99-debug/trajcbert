#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --job-name=trajcbert_test_with_context_1_2_data_bs_32_20_epochs_2_time
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

python testing_1_point_with_context_2_test_on_20_epochs_1_2_data.py


