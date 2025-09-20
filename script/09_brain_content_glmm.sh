#!/bin/bash
#SBATCH --job-name=brain_content_glmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/brain_content_glmm-%j.out
#SBATCH --error=../logs/brain_content_glmm-%j.err
#SBATCH --array=0-74  # 75 jobs total: 5 thresholds × 5 clusters × 3 data types
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

idx=$SLURM_ARRAY_TASK_ID  # SLURM arrays are 0-indexed
num_clusters=5
num_datatypes=3  # combined, paired, balanced

# Extract parameters
threshold_idx=$((idx / (num_clusters * num_datatypes)))
remainder=$((idx % (num_clusters * num_datatypes)))
datatype_idx=$((remainder / num_clusters))
cluster_idx=$((remainder % num_clusters))

# Define parameter arrays
thresholds=(0.60 0.65 0.70 0.75 0.80)
data_types=("combined" "paired" "balanced")
cluster_ids=(1 2 3 4 5)

# Get actual parameter values
THRESHOLD=${thresholds[$threshold_idx]}
DATA_TYPE=${data_types[$datatype_idx]}
CLUSTER_ID=${cluster_ids[$cluster_idx]}

# Function to get model pattern based on cluster mapping from script 07
# You may need to adjust these based on your actual cluster-to-state mapping
get_model_pattern() {
    local cluster=$1
    local threshold=$2

    # Default patterns based on common configurations
    # These should match what script 07 actually produces
    case $cluster in
        1) echo "2states" ;;
        2) echo "2states" ;;
        3) echo "3states" ;;
        4) echo "10states" ;;
        5) echo "12states" ;;
        *) echo "2states" ;;
    esac
}

# Get model patterns
MODEL_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD)

echo "Running with: threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID, model_pattern=$MODEL_PATTERN"

# Run the appropriate command based on data type
if [ "$DATA_TYPE" == "combined" ]; then
    echo "Running combined analysis..."
    python 09_brain_content_glmm.py \
        --threshold $THRESHOLD \
        --data_type combined \
        --cluster_id $CLUSTER_ID \
        --model_pattern $MODEL_PATTERN \
        --combined_group combined

elif [ "$DATA_TYPE" == "balanced" ]; then
    echo "Running balanced analysis..."
    python 09_brain_content_glmm.py \
        --threshold $THRESHOLD \
        --data_type combined \
        --cluster_id $CLUSTER_ID \
        --model_pattern $MODEL_PATTERN \
        --combined_group balanced

elif [ "$DATA_TYPE" == "paired" ]; then
    echo "Running paired analysis..."
    # For paired, we need both affair and paranoia model patterns
    # They might be the same or different depending on your design
    AFFAIR_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD)
    PARANOIA_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD)

    python 09_brain_content_glmm.py \
        --threshold $THRESHOLD \
        --data_type paired \
        --cluster_id $CLUSTER_ID \
        --affair_model_pattern $AFFAIR_PATTERN \
        --paranoia_model_pattern $PARANOIA_PATTERN
fi

echo "Job completed for threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID"