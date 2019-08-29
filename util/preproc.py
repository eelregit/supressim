import os, sys

import numpy as np

from bigfile import File


#def get_grid(pid, boxsize, Ng):
#    """Assume `pid` starts from 0 and aligns with the Lagrangian lattice.
#    """
#    cellsize = boxsize / Ng
#    z = pid % Ng
#    y = pid // Ng % Ng
#    x = pid // (Ng * Ng)
#    grid = np.stack([x, y, z], axis=-1)
#    grid *= cellsize
#    return grid


def pos2dis(pos, boxsize, Ng):
    """Assume `pos` is ordered in `pid` that aligns with the Lagrangian lattice,
    and all displacement must not exceed half box size.
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize

    pos[..., 0] -= lattice.reshape(-1, 1, 1)
    pos[..., 1] -= lattice.reshape(-1, 1)
    pos[..., 2] -= lattice

    pos -= np.rint(pos / boxsize) * boxsize

    return pos


def set_subgrids(Ng):
    Nsg = Ng // 5
    offset = Nsg // 2
    return Nsg, offset


def save_subgrids(outpath, x, Ng, Nsg, offset):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    starts = range(0, Ng, offset)

    for i, i_start in enumerate(starts):
        i_stop = i_start + Nsg
        i_ind = np.arange(i_start, i_stop).reshape(-1, 1, 1)
        i_ind %= Ng
        for j, j_start in enumerate(starts):
            j_stop = j_start + Nsg
            j_ind = np.arange(j_start, j_stop).reshape(-1, 1)
            j_ind %= Ng
            for k, k_start in enumerate(starts):
                k_stop = k_start + Nsg
                k_ind = np.arange(k_start, k_stop)
                k_ind %= Ng
                np.save(outpath + "{}{}{}.npy".format(i, j, k), x[i_ind, j_ind, k_ind])


def get_nonlin_fields(inpath, outpath):

    bigf = File(inpath)

    header = bigf.open('Header')
    boxsize = header.attrs['BoxSize'][0]
    Ng = header.attrs['TotNumPart'][1] ** (1/3)
    Ng = int(np.rint(Ng))

    cellsize = boxsize / Ng

    pid_ = bigf.open('1/ID')[:] - 1  # so that particle id starts from 0
    pos_ = bigf.open('1/Position')[:]
    pos = np.empty_like(pos_)
    pos[pid_] = pos_
    pos = pos.reshape(Ng, Ng, Ng, 3)
    vel_ = bigf.open('1/Velocity')[:]
    vel = np.empty_like(vel_)
    vel[pid_] = vel_
    vel = vel.reshape(Ng, Ng, Ng, 3)
    del pid_, pos_, vel_

    dis = pos2dis(pos, boxsize, Ng)
    del pos

    dis = dis.astype('f4')
    vel = vel.astype('f4')
    fields = np.concatenate([dis, vel], axis=-1)
    del dis, vel

    Nsg, offset = set_subgrids(Ng)

    save_subgrids(outpath, fields, Ng, Nsg, offset)


if __name__ == '__main__':
    sim_start, sim_stop, snap_start, snap_stop = sys.argv[1:]
    sims = range(int(sim_start), int(sim_stop))
    snaps = range(int(snap_start), int(snap_stop))
    snaps = ["{:03d}".format(snap) for snap in snaps]

    redshifts = {'001': 9.0,
                 '002': 8.0,
                 '003': 7.5,
                 '004': 7.0,
                 '005': 6.5,
                 '006': 6.0,
                 '007': 5.5,
                 '008': 5.0,
                 '009': 4.5,
                 '010': 4.0,
                 '011': 3.5,
                 '012': 3.0}

    # path
    subpath_lores = "low-resl/set{}/output/PART_{}/"
    subpath_hires = "high-resl/set{}/output/PART_{}/"

    inpath_root = "/scratch1/06431/yueyingn/dmo-50MPC-fixvel/"
    outpath_root = "/scratch1/06589/yinli/dmo-50MPC-fixvel/"

    for sim in sims:
        for snap in snaps:
            for subpath in [subpath_lores, subpath_hires]:
                inpath = inpath_root + subpath.format(sim, snap)
                outpath = outpath_root + subpath.format(sim, snap)
                get_nonlin_fields(inpath, outpath)
