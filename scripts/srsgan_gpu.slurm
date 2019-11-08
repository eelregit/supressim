#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:4

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

hostname; pwd; date

BATCH_SIZE=$SLURM_CPUS_PER_TASK

srsgan.py --batch-size $BATCH_SIZE \
          --n-cpu $BATCH_SIZE \
          --epoch 11 \
          --lambda-content 1000 \
          --lambda-adversarial 0 \
          --lr-G 0.00005 \
          --lr-D 0 \
          --checkpoint-interval 1 \
          --hr-glob-path '/mnt/ceph/users/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy'

date
