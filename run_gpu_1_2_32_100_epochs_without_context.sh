#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:4
#SBATCH --time=7-0:00
#SBATCH --job-name=trajcbert_on_gpu_1_2_batch_size_32_4_DAYS_100_epochs_without_context
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=16 # number of cores for each task
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca




module load scipy-stack/2023a
VENV_DIR=$SLURM_TMPDIR/MYENV
virtualenv  $VENV_DIR
source $VENV_DIR/bin/activate
pip install -r requirements.txt --no-index
pip list

python parallelisation_gpu_train_1_2_bs_32_100_epochs_without_context.py


