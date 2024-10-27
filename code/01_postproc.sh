#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH --partition=mit_normal
#SBATCH --output=../logs/postproc-%j.out
#SBATCH --error=../logs/postproc-%j.err
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=yibei@mit.edu

source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate prettymouth

sub_ids=("sub-023" "sub-030" "sub-032" "sub-034" "sub-038" "sub-050" "sub-052" "sub-065" "sub-066" "sub-079" "sub-081" "sub-083" "sub-084" "sub-085" "sub-086" "sub-087" "sub-088" "sub-089" "sub-090" "sub-091" "sub-092" "sub-093" "sub-094" "sub-095" "sub-096" "sub-097" "sub-098" "sub-099" "sub-100" "sub-101" "sub-102" "sub-103" "sub-104" "sub-105" "sub-106" "sub-107" "sub-108" "sub-109" "sub-110" "sub-111")

SUB_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

python 01_postproc.py $SUB_ID