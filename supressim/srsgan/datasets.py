from glob import glob
import numpy as np
from scipy.special import hyp2f1
import torch
from torch.utils.data import Dataset


class BoxesDataset(Dataset):
    def __init__(self, hr_glob_path, augment=True):
        self.hr_files = sorted(glob(hr_glob_path))
        self.lr_files = [hr_file.replace("high", "low", 1) for hr_file in self.hr_files]

        self.augment = augment

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        lr_box = np.load(self.lr_files[idx])
        hr_box = np.load(self.hr_files[idx])

        # channels, the phase-space coords, need to be at front
        lr_box = np.moveaxis(lr_box, -1, 0)
        hr_box = np.moveaxis(hr_box, -1, 0)

        if self.augment:
            lr_box, hr_box = _random_flip_3d(lr_box, hr_box)
            lr_box, hr_box = _random_permute_3d(lr_box, hr_box)

        lr_box = lr_box[:3]
        hr_box = hr_box[:3]

        normalize(lr_box)
        normalize(hr_box)

        lr_box = torch.from_numpy(lr_box).float()
        hr_box = torch.from_numpy(hr_box).float()

        return lr_box, hr_box


def _random_flip_3d(*arrays):
    axes = np.random.randint(2, size=3, dtype=bool)
    axes = np.arange(3)[axes]

    axes_twice = np.concatenate([axes, 3 + axes])
    axes = 1 + axes

    flipped_arrays = []
    for a in arrays:
        a[axes_twice] = - a[axes_twice]  # flip vector components
        a = np.flip(a, axis=axes)
        flipped_arrays.append(a)

    return flipped_arrays

def _random_permute_3d(*arrays):
    axes = np.random.permutation(3)

    axes_twice = np.concatenate([axes, 3 + axes])
    axes = np.insert(1 + axes, 0, 0)

    permuted_arrays = []
    for a in arrays:
        a = a[axes_twice]  # permutate vector components
        a = a.transpose(axes)
        permuted_arrays.append(a)

    return permuted_arrays


def normalize(a, reverse=False):
    z = 7  # FIXME
    dis_norm = 6000 * D(z)  # [kpc/h]
    # for dmo, vel is actually the canonical momentum $a v = a^2 \dot x$
    vel_norm = dis_norm * H(z) * f(z) / (1 + z)**2  # [km/s]

    if not reverse:
        dis_norm = 1 / dis_norm
        vel_norm = 1 / vel_norm

    a[:3] *= dis_norm
    #a[3:] *= vel_norm


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return a * hyp2f1(1, 1/3, 11/6, - OL * a**3 / Om) \
             / hyp2f1(1, 1/3, 11/6, - OL / Om)

def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    aa3 = OL * a**3 / Om
    return 1 - 6/11*aa3 * hyp2f1(2, 4/3, 17/6, -aa3) \
                        / hyp2f1(1, 1/3, 11/6, -aa3)

def H(z, Om=0.31):
    """Hubble in [h km/s/kpc] for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return 100 * np.sqrt(Om / a**3 + OL) * 1e-3
