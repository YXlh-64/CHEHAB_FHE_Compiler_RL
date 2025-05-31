#!/bin/bash

#SBATCH -p nvidia              # Use the nvidia partition
#SBATCH -q c2                  # Specify the queue c2
#SBATCH --gres=gpu:a100:3   # Request 3 A100 GPU
#SBATCH --mem=350G
#SBATCH -t 10-1                # Runtime of 10 days
#SBATCH --output=/scratch/ad7786/chehab-vectorization-rl/stdout/output_%j.out
#SBATCH --error=/scratch/ad7786/chehab-vectorization-rl/stderr/err_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ad7786@nyu.edu
#SBATCH --job-name=chehab-vectorization-embeddings-5m-15depth-32vecsize

module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate main_env

#python3 /scratch/ad7786/chehab-vectorization-rl/embeddings_AETR_CLS_single.py train
torchrun --nproc_per_node=3 /scratch/ad7786/chehab-vectorization-rl/embeddings_AETR_CLS.py train #--snapshot --profile 
