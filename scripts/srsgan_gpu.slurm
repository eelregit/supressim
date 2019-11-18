#!/bin/bash

#SBATCH --job-name=srsgan
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-32gb:4

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=1-00:00:00

hostname; pwd; date

source $HOME/anaconda3/bin/activate torch

module load gcc openmpi2
module load cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1

BATCH_SIZE=8

srsgan.py --batch-size $BATCH_SIZE \
          --num-workers $BATCH_SIZE \
          --epoch 0 \
          --lambda-content 1000 \
          --lambda-adversarial 1 \
          --lr-G 0.0004 \
          --lr-D 0.0001 \
          --checkpoint-interval 1 \
          --hr-glob-path '/mnt/ceph/users/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy'

date
