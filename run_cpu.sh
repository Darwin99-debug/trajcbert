#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --time=03:00:00 
time = $(date +"%T")
#SBATCH --job-name=first_small_trajcbert_on_cpu_s_and_venv+$time
#SBATCH --output=outputs/%x-%j-$time.out
#SBATCH --error=errors/%x-%j-$time.err
#SBATCH --cpus-per-task=39 # number of cores
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=3 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca






module load python/3.10.2 scipy-stack/2023a

python first_test_small_train.py


