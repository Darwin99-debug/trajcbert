#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --time=03:00:00 
#SBATCH --job-name=first_small_trajcbert_on_cpu_s_and_venv
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=39 # number of cores
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=3 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca





module load scipy-stack/2023a
VENV_DIR=$SLURM_TMPDIR/MYENV
virtualenv  $VENV_DIR
source $VENV_DIR/bin/activate
pip install scikit-build --no-index
pip install flake8 --no-index
pip install h3
pip install -r requirements.txt --no-index
pip list
python first_test_small_train.py


