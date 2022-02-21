#!/bin/bash
#
#SBATCH --job-name=deteksi
#SBATCH --output=logs/deteksi_%A.out
#SBATCH --error=logs/deteksi_%A.err
#
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=komputasi08

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/m450296/miniconda3/envs/text/lib
source ~/miniconda3/etc/profile.d/conda.sh
conda activate text

python kandidat_MSRA.py
