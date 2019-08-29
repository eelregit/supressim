#!/bin/bash

#SBATCH --job-name=srgan
#SBATCH --output=%x-%j.out

#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00

hostname; pwd; date

export OMP_NUM_THREADS=56

python srgan.py

date
