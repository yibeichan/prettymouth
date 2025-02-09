#!/bin/bash
#SBATCH --job-name=group_hmm
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/group_hmm-%j.out
#SBATCH --error=../logs/group_hmm-%j.err
#SBATCH --array=0-37
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

# Activate your Conda environment
source ~/.bashrc  # or source /etc/profile.d/conda.sh
micromamba activate prettymouth

# Arrays
group_names=("affair" "paranoia")
n_states=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Generate combinations
combinations=()
for group in "${group_names[@]}"; do
    for state in "${n_states[@]}"; do
        combinations+=("$group $state")
    done
done

# Ensure SLURM_ARRAY_TASK_ID is valid
if [ -z "$SLURM_ARRAY_TASK_ID" ] || [ "$SLURM_ARRAY_TASK_ID" -ge "${#combinations[@]}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set or out of range."
    exit 1
fi

# Extract the combination based on SLURM_ARRAY_TASK_ID
selected_combination="${combinations[$SLURM_ARRAY_TASK_ID]}"
group_name=$(echo "$selected_combination" | awk '{print $1}')
n_state=$(echo "$selected_combination" | awk '{print $2}')

# Debugging info
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Selected group_name: $group_name"
echo "Selected n_states: $n_state"

# Run Python script
python 06_single_group_hmm_ntw.py --group_name "$group_name" --n_states "$n_state"