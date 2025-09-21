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

# Function to get model pattern based on actual available directories
get_model_pattern() {
    local cluster=$1
    local threshold=$2
    local data_type=$3

    # Format threshold for directory naming (e.g., 0.80 -> 080)
    local thresh_fmt=$(printf "%.2f" $threshold | tr -d '.')

    # Base path for cluster directories
    local base_path="/orcd/scratch/bcs/001/yibei/prettymouth/output_RR/07_map_cluster2stateseq"

    # Find directories matching the pattern for this cluster
    # Pattern: th_{threshold}_{data_type}_*states_cluster{cluster_id}
    local pattern="th_${thresh_fmt}_${data_type}_*states_cluster${cluster}"

    # Find the matching directory
    local found_dir=$(ls -d ${base_path}/${pattern} 2>/dev/null | head -1)

    if [ -z "$found_dir" ]; then
        echo "ERROR: No directory found for pattern ${pattern}" >&2
        echo "2states"  # Default fallback
        return 1
    fi

    # Extract the states part from the directory name
    # Format: th_080_combined_7states_cluster2 -> 7states
    local model_pattern=$(basename "$found_dir" | sed -E 's/.*_([0-9]+states)_cluster.*/\1/')

    echo "$model_pattern"
}

# Get model patterns
MODEL_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD $DATA_TYPE)
pattern_status=$?

echo "Running with: threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID, model_pattern=$MODEL_PATTERN"

# Check if the model pattern was found (get_model_pattern returns 1 on error)
if [ $pattern_status -eq 1 ]; then
    echo "WARNING: Skipping non-existent combination: threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID"
    echo "No matching directory found in filesystem"
    exit 0
fi

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
    AFFAIR_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD "affair")
    PARANOIA_PATTERN=$(get_model_pattern $CLUSTER_ID $THRESHOLD "paranoia")

    # Check if both patterns were found
    affair_status=$?
    paranoia_status=$?

    if [ $affair_status -eq 1 ] || [ $paranoia_status -eq 1 ]; then
        echo "WARNING: Skipping paired analysis - missing affair or paranoia data"
        echo "Affair pattern: $AFFAIR_PATTERN, Paranoia pattern: $PARANOIA_PATTERN"
        exit 0
    fi

    python 09_brain_content_glmm.py \
        --threshold $THRESHOLD \
        --data_type paired \
        --cluster_id $CLUSTER_ID \
        --affair_model_pattern $AFFAIR_PATTERN \
        --paranoia_model_pattern $PARANOIA_PATTERN
fi

echo "Job completed for threshold=$THRESHOLD, data_type=$DATA_TYPE, cluster_id=$CLUSTER_ID"