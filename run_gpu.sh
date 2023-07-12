#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00 
time = $(date +"%T")
#SBATCH --job-name=first_small_trajcbert_on_gpu_s_and_venv-$time
#SBATCH --output=outputs/%x-%j-$time.out
#SBATCH --error=errors/%x-%j-$time.err
#SBATCH --cpus-per-task=16 # number of cores
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca





source /home/daril/scratch/MYENV/bin/activate
module scipy-stack/2023a


python first_test_small_train_gpu.py


