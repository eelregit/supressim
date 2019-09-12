#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out

#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00

hostname; pwd; date

export OMP_NUM_THREADS=56

LOG_DIR=$SCRATCH/slurm_log
LOG_NAME=$SLURM_JOB_NAME-$SLURM_JOB_ID.out
LOG_PATH=$LOG_DIR/$LOG_NAME

mkdir -p "$LOG_DIR"

srsgan.py --checkpoint-interval 1 &> "$LOG_PATH"

echo stdout and stderr have been redirected to $LOG_PATH

date
