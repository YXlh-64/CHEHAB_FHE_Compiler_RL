#!/bin/bash

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 5-0
#SBATCH --output=/scratch/bs5331/chehab-vectorization-rl/stdout/output_%j.out
#SBATCH --error=/scratch/bs5331/chehab-vectorization-rl/stderr/err_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs5331@nyu.edu
#SBATCH --job-name=chehab-vectorization-rl
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"
conda activate main_env

# Pass the model configuration file
#export OPENAI_API_KEY=
#python /scratch/bs5331/chehab-vectorization-rl/llm_data_generator.py --vec-size=4 --iterations=500 --output=./vec_lang_expressions_gpt_vec_4.txt
python3 /scratch/bs5331/chehab-vectorization-rl/model6.py train
