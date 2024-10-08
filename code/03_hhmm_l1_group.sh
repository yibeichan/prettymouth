#!/bin/bash
#SBATCH --job-name=hhmm_l1_group
#SBATCH --output=../logs/hhmm_l1_group-%j.out
#SBATCH --error=../logs/hhmm_l1_group-%j.err
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate prettymouth2
networks=('Aud' 'ContA' 'ContB' 'ContC' 'DefaultA' 'DefaultB' 'DefaultC' 'DorsAttnA' 'DorsAttnB' 'Language' 'SalVenAttnA' 'SalVenAttnB' 'SomMotA' 'SomMotB' 'VisualA' 'VisualB' 'VisualC' 'Subcortical')
groups=('affair' 'paranoia')

# Calculate the network and group indices
network_index=$((SLURM_ARRAY_TASK_ID / ${#groups[@]}))
group_index=$((SLURM_ARRAY_TASK_ID % ${#groups[@]}))

network=${networks[$network_index]}
group=${groups[$group_index]}

python 03_hhmm_l1_group.py --network $network --group $group
