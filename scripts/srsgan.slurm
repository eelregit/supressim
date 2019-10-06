#!/bin/bash

#SBATCH --job-name=srgan
#SBATCH --output=%x-%j.out
#SBATCH --partition=normal
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=2
##SBATCH --ntasks=2
#SBATCH --time=1-00:00:00
#SBATCH --exclusive

hostname; pwd; date

export OMP_NUM_THREADS=56

NPROCS_PER_NODE=1

# Here batch-size is batch size per node when applying DDP on cpu module
COMMAND="/home1/06431/yueyingn/test/test-0929/supressim/srsgan.py  --batch-size 2"

#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"

MPORT="3355"

echo $HOSTLIST
#Launch the pytorch processes, first on master (first in $HOSTLIST) then on the slaves

RANK=0
for node in $HOSTLIST; do
        ssh -q $node \
		python -m torch.distributed.launch \
		--nproc_per_node=$NPROCS_PER_NODE \
		--nnodes=$SLURM_JOB_NUM_NODES \
		--node_rank=$RANK \
		--master_addr="$MASTER" --master_port="$MPORT" \
		$COMMAND &
	RANK=$((RANK+1))
done
wait
