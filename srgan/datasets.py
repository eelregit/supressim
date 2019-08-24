from glob import glob
import numpy as np
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
            pass # TODO: data augmentation

        lr_box = torch.from_numpy(lr_box).float()
        hr_box = torch.from_numpy(hr_box).float()

        return lr_box, hr_box
