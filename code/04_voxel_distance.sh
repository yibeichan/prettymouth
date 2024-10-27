#!/bin/bash

#SBATCH --job-name=wholebrain_distance
#SBATCH --output=../logs/wholebrain_distance_%j.out
#SBATCH --error=../logs/wholebrain_distance_%j.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate prettymouth

python 04_voxel_distance.py