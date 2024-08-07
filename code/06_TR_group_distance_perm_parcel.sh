#!/bin/bash
#SBATCH --job-name=TR_group_distance_perm
#SBATCH --output=../logs/TR_group_distance_%j.out
#SBATCH --error=../logs/TR_group_distance_%j.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gablab
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

python 06_TR_group_distance_perm_parcel.py $n_parcel
