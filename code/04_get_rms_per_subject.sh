#!/bin/bash
#SBATCH --job-name=get_RMS
#SBATCH --output=../logs/get_RMS_%j.out
#SBATCH --error=../logs/get_RMS_%j.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

types=("inter_coflt" "intra_coflt")
# n_parcels=(100 200 300 400 500 600 700 800 900 1000)
n_parcels=(100 200 300 400)

task_id=$((SLURM_ARRAY_TASK_ID % ${#types[@]}))
parcel_id=$((SLURM_ARRAY_TASK_ID / ${#types[@]}))

cal_type=${types[task_id]}
n_parcel=${n_parcels[parcel_id]}

echo "cal_type: $types"
echo "n_parcel: $n_parcel"

python 04_get_rms_per_subject.py $n_parcel $cal_type
