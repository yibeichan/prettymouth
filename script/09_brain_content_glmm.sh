#!/bin/bash
#SBATCH --job-name=brain_content_glmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/brain_content_glmm-%j.out
#SBATCH --error=../logs/brain_content_glmm-%j.err
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

idx=$((SLURM_ARRAY_TASK_ID - 1))
num_clusters=5
num_datatypes=2

# Extract parameters
threshold_idx=$((idx / (num_clusters * num_datatypes)))
remainder=$((idx % (num_clusters * num_datatypes)))
datatype_idx=$((remainder / num_clusters))
cluster_idx=$((remainder % num_clusters))

# Define parameter arrays
thresholds=(0.60 0.65 0.70 0.75 0.80)
data_types=("combined" "paired")
cluster_ids=(1 2 3 4 5)

# Get actual parameter values
THRESHOLD=${thresholds[$threshold_idx]}
DATA_TYPE=${data_types[$datatype_idx]}
CLUSTER_ID=${cluster_ids[$cluster_idx]}

echo "Running with: threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID"

python 09_brain_content_glmm.py --threshold $THRESHOLD --data_type $DATA_TYPE --cluster_id $CLUSTER_ID