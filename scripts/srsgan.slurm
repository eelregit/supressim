#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out

#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00

hostname; pwd; date

export OMP_NUM_THREADS=56

LOG_PATH=$PWD/$SLURM_JOB_NAME-$SLURM_JOB_ID.out

srsgan.py --checkpoint-interval 1 &>> "$LOG_PATH"

date
