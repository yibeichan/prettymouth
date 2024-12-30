#!/bin/bash
#SBATCH --job-name=group_hmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/group_hmm-%j.out
#SBATCH --error=../logs/group_hmm-%j.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Activate your Conda environment
source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

python 05_group_hmm_ntw_find_n_states.py