#!/bin/bash
#SBATCH --job-name=group_hmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/group_hmm-%j.out
#SBATCH --error=../logs/group_hmm-%j.err
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=36G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu


# Activate your Conda environment
source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

groups=("combined" "affair" "paranoia")
group=${groups[$SLURM_ARRAY_TASK_ID]}
python 05_group_hmm.py --group "$group"
