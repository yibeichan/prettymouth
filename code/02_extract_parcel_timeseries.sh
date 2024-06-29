#!/bin/bash
#SBATCH --job-name=extract_parcel_ts
#SBATCH --output=../logs/extract_parcel_ts_%j.out
#SBATCH --error=../logs/extract_parcel_ts_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-10
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

# Map array index to n_parcel value
declare -a values=(100 200 300 400 500 600 700 800 900 1000)

n_parcel=${values[$SLURM_ARRAY_TASK_ID - 1]}

python 02_extract_parcel_timeseries.py --n_parcel $n_parcel
