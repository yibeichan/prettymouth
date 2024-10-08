#!/bin/bash
#SBATCH --job-name=hhmm_l1_all
#SBATCH --output=../logs/hhmm_l1_all-%j.out
#SBATCH --error=../logs/hhmm_l1_all-%j.err
#SBATCH --array=0-17
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

network=${networks[$SLURM_ARRAY_TASK_ID]}

python 03_hhmm_l1_all.py --network $network
