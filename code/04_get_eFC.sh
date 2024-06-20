#!/bin/bash
#SBATCH --job-name=get_eFC
#SBATCH --output=../logs/get_eFC_%j.out
#SBATCH --error=../logs/get_eFC_%j.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

# n_parcels=(100 200 300 400)

n_parcels=(400)

n_parcel=${n_parcels[$SLURM_ARRAY_TASK_ID]}

echo "n_parcel: $n_parcel"

python 04_get_eFC.py $n_parcel
