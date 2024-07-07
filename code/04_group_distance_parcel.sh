#!/bin/bash
#SBATCH --job-name=group_distance
#SBATCH --output=../logs/group_distance_%j.out
#SBATCH --error=../logs/group_distance_%j.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
source activate prettymouth

n_parcels=(800 900)

n_parcel=${n_parcels[$SLURM_ARRAY_TASK_ID]}

echo "n_parcel: $n_parcel"

python 04_group_distance_parcel.py $n_parcel
