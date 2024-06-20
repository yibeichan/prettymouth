#!/bin/bash
#SBATCH --job-name=get_sim_mtx
#SBATCH --output=../logs/get_sim_mtx_%j.out
#SBATCH --error=../logs/get_sim_mtx_%j.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

# n_parcels=(100 200 300 400)

# Define the possible values
n_parcels=(400)
eFC_types=("roi_network" "network")
groups=("affair" "paranoia")

# Compute the index for each variable based on SLURM_ARRAY_TASK_ID
index_n_parcel=$((SLURM_ARRAY_TASK_ID / (2 * 2) % 1))
index_eFC_type=$((SLURM_ARRAY_TASK_ID / 2 % 2))
index_group=$((SLURM_ARRAY_TASK_ID % 2))

# Get the value for each variable
n_parcel=${n_parcels[$index_n_parcel]}
eFC_type=${eFC_types[$index_eFC_type]}
group=${groups[$index_group]}

echo "n_parcel: $n_parcel"
echo "eFC_type: $eFC_type"
echo "group: $group"

python 07_get_sim_mtx.py $n_parcel $eFC_type $group