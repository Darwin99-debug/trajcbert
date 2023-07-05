#!/usr/bin/env bash
#SBATCH --account=def-nkambou
#SBATCH --time=0-00:3:00 # time (DD-HH:MM)
#SBATCH --job-name=first_small_trajcbert_on_cpu_s
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=39 # number of cores
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=3 
#SBATCH --mail-type=ALL
#SBATCH --mail-user= lacourarie.clara@courrier.uqam.ca



VENV_DIR=venv

source "$VENV_DIR"/bin/activate

# python3 -m pip install --upgrade pip



#  run the model located in mode.py
python3 first_test_small_train.py
# this code is to lunch the model on calcul quebec
