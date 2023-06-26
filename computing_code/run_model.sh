#!/usr/bin/env bash
#SBATCH --account=def-nkambou
#SBATCH --time=0-00:10:00 # time (DD-HH:MM)
#SBATCH --job-name=test_installation
#SBATCH --output=output.log
#SBATCH --error=error.log



VENV_DIR=venv

# creation of the virtual environment
python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR"/bin/activate
pip install -r requirements.txt


#  run the model located in mode.py
python model.py

# this code is to lunch the model on calcul quebec