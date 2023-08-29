#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:4
#SBATCH --time=4-0:00
#SBATCH --job-name=trajcbert_on_gpu_full_batch_size_16_4_DAYS
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=16 # number of cores
#SBATCH --mem=0 
#SBATCH --nodes=1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca




module load scipy-stack/2023a
VENV_DIR=$SLURM_TMPDIR/MYENV
virtualenv  $VENV_DIR
source $VENV_DIR/bin/activate
pip install -r requirements.txt --no-index
pip list

python parallelisation_gpu_train_full.py


