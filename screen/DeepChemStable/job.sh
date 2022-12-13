#!/bin/sh
#An example for gpu job.
#SBATCH -J 15check
#SBATCH --output=15check.out
#SBATCH --error=15check.err
#SBATCH -p gpuQ 
source activate tfgpu

python regression.py
