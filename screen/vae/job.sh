#!/bin/sh
#An example for gpu job.
#SBATCH -J vae
#SBATCH --output=gatmmm5.out
#SBATCH --error=gatmmm5.err
#SBATCH -p gpu4Q --gres=gpu:2
source activate torch

python vae.py
