#!/bin/bash
#SBATCH --account=def-nkambou
#SBATCH --gres=gpu:4
#SBATCH --time=4-0:00
#SBATCH --job-name=trajcbert_on_gpu_1_2_batch_size_32_4_DAYS_20_epochs
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err
#SBATCH --cpus-per-task=16 # number of cores for each task
#SBATCH --nodes=3 
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kengne_wambo.daril_raoul@courrier.uqam.ca

nodes = ( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array = ( $nodes )
head_node = ${nodes_array[0]}
head_node_ip = $(getent hosts $head_node | awk '{ print $1 }')
echo "head node: $head_node"
echo "head node ip: $head_node_ip"
echo "nodes: $nodes"
export LOGLEVEL=INFO




module load scipy-stack/2023a
VENV_DIR=$SLURM_TMPDIR/MYENV
virtualenv  $VENV_DIR
source $VENV_DIR/bin/activate
pip install -r requirements.txt --no-index
pip list

srun torchrun --nodelist=$head_node --nnodes=1 --nproc_per_node=1 python parallelisation_gpu_train_1_2_bs_32_20_epochs.py & \  

python parallelisation_gpu_train_1_2_bs_32_20_epochs.py


