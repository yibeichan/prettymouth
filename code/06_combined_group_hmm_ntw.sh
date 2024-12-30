#!/bin/bash
#SBATCH --job-name=group_hmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/group_hmm-%j.out
#SBATCH --error=../logs/group_hmm-%j.err
#SBATCH --array=0-18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Activate your Conda environment
source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

n_states=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

n_state=${n_states[$SLURM_ARRAY_TASK_ID]}
# Run Python script
python 06_combined_group_hmm_ntw.py  --n_states "$n_state" 