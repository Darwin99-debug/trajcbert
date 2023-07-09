#!/usr/bin/env bash
#SBATCH --account=def-nkambou
// request gpu 
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:3:00 # time (DD-HH:MM)
#SBATCH --job-name=first_small_trajcbert_on_cpu_s_and_venv
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=39 # number of cores
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=3 
#SBATCH --mail-type=ALL
#SBATCH --mail-user= kengne_wambo.daril_raoul@courrier.uqam.ca




VENV_DIR=/home/daril/scratch/MYENV

source "$VENV_DIR"/bin/activate

# python3 -m pip install --upgrade pip



#  run the model located in mode.py
python3 first_test_small_train.py
# this code is to lunch the model on calcul quebec
