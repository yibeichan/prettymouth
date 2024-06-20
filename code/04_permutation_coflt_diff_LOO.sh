#!/bin/bash
#SBATCH --job-name=perm_coflt_diff_LOO
#SBATCH --output=../logs/perm_coflt_diff_LOO_%j.out
#SBATCH --error=../logs/perm_coflt_diff_LOO_%j.err
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Load necessary modules
source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/singularity/3.10.4
source activate prettymouth

n_parcels=(400 500 600 700 800 900 1000)

n_parcel=${n_parcels[$SLURM_ARRAY_TASK_ID]}

echo "n_parcel: $n_parcel"

python 04_permutation_coflt_diff_LOO.py $n_parcel
