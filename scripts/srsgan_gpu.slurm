#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00

hostname; pwd; date

BATCH_SIZE=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$BATCH_SIZE

srsgan.py --batch-size $BATCH_SIZE \
          --n-cpu $BATCH_SIZE \
          --checkpoint-interval 1 \
          --hr-glob-path '/mnt/ceph/users/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy'

date
