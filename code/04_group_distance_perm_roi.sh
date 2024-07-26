#!/bin/bash
#SBATCH --job-name=group_distance_perm_roi
#SBATCH --output=../logs/group_distance_%j.out
#SBATCH --error=../logs/group_distance_%j.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
source ~/.bashrc
conda activate prettymouth2

# n_parcels=(500 600 700 800 900 1000)
n_parcels=(400)

n_parcel=${n_parcels[$SLURM_ARRAY_TASK_ID]}

echo "n_parcel: $n_parcel"

python 04_group_distance_perm_roi.py $n_parcel
