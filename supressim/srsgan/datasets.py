from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset


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


        lr_box = torch.from_numpy(lr_box).float()
        hr_box = torch.from_numpy(hr_box).float()

        return lr_box, hr_box
