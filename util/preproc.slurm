#!/bin/bash

#SBATCH --job-name=preproc
#SBATCH --output=%x-%j.out

#SBATCH --partition=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-01:00:00

hostname; pwd; date

./preproc.py 0 5 004 005

date
