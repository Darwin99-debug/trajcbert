#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --time=0-00:10:00 # time (DD-HH:MM)


# creation of the virtual environment
virtualenv -p python3.6 venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt


#  run the model located in mode.py
python model.py

# this code is to lunch the model on calcul quebec