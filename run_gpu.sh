#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00 
#SBATCH --job-name=first_small_trajcbert_on_gpu_s_and_venv
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=16 # number of cores
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca





module load python/3.9.6 StdEnv/2020 scipy-stack/2023a




python first_test_small_train_gpu.py


