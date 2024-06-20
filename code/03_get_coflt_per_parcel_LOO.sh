#!/bin/bash
#SBATCH --job-name=get_coflt_per_parcel
#SBATCH --output=../logs/get_coflt_per_parcel-%j.out
#SBATCH --error=../logs/get_coflt_per_parcel-%j.err
#SBATCH --array=0-7 # 2 groups * 4 parcels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

groups=("affair" "paranoia")
# n_parcels=(100 200 300 400 500 600 700 800 900 1000)
n_parcels=(100 200 300 400)

task_id=$((SLURM_ARRAY_TASK_ID % ${#groups[@]}))
parcel_id=$((SLURM_ARRAY_TASK_ID / ${#groups[@]}))

group_id=${groups[task_id]}
n_parcel=${n_parcels[parcel_id]}

echo "group_id: $group_id"
echo "n_parcel: $n_parcel"

python 03_get_coflt_per_parcel_LOO.py $group_id $n_parcel