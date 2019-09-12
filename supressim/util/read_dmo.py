"""Example to read the particle data using bigfile library.

DMO sims by Yueying:

The multi-resolution 50MPC-DM-only simulations are in
`stampede2:/work/06431/yueyingn/mysharedirectory/dmo-50MPC-fixvel/`

There are 5 sets of simulation (different IC realizations) run in high
resolution (880^3 particles in 50MPC) and low resolution (440^3 particles in
50MPC),  stored in `high-resl/set*/output and /low-resl/set*/output`

The output/Snapshot.txt list out the redshift of the snapshots: set0 has run
down to z=3, and all other sets only run down to z=7 (I am not sure to which
redshift do we need, so I stopped at z=7 for later data sets --- I can resume
the run quickly if you need lower redshift snapshots.)
"""

import numpy as np
from bigfile import File, Dataset

# list all the quantities
dm_quantity = ['1/GroupID','1/ID','1/Mass','1/Position','1/Velocity']

fof_quantity = ['FOFGroups/FirstPos','FOFGroups/GroupID','FOFGroups/Imom','FOFGroups/Jmom',
                'FOFGroups/LengthByType','FOFGroups/Mass','FOFGroups/MassByType',
                'FOFGroups/MassCenterPosition','FOFGroups/MassCenterVelocity','FOFGroups/MinID']

snapshot = "/work/06431/yueyingn/mysharedirectory/dmo-50MPC-fixvel/low-resl/set0/output/PART_004"
pig = File(snapshot)

# redshift
redshift = 1. / pig.open('Header').attrs['Time'] - 1
print ('z=', redshift)

# Particle information:
DM_id = pig.open('1/ID')[:] - 1
DM_pos = pig.open('1/Position')[:]  # in kpc
DM_vel = pig.open('1/Velocity')[:]

DM = Dataset(pig["1/"], ["ID", "Position", "Velocity"])
