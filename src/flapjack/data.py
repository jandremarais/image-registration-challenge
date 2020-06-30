# import random
# from functools import partial
from pathlib import Path
# from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
# from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.transforms import functional as TF


class Misaligned(Dataset):
    def __init__(self, path: Path, sz: int = 128, mean=(0, 0, 0, 0), std=(1, 1, 1, 1)):
        super().__init__()
        self.image_fns = list((path / "images").iterdir())
        self.target_fns = [
            path / f"targets/target{o.stem.lstrip('image')}.npy" for o in self.image_fns
        ]
        self.sz = sz
        self.mean = mean
        self.std = std

    def transform(self, x):
        rgb = x[..., :3].astype(np.uint8)
        red = x[..., 3]

        rgb_tfms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.sz),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[:3], self.std[:3]),
            ]
        )

        red_tfms = transforms.Compose(
            [
                transforms.ToPILImage(mode="F"),
                transforms.Resize(self.sz),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[3], self.std[3]),
            ]
        )
        return torch.cat([rgb_tfms(rgb), red_tfms(red)])

    def __getitem__(self, index):
        x = np.load(self.image_fns[index])
        y = np.load(self.target_fns[index])
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.image_fns)

    @staticmethod
    def plot(x, y=None, sidebyside=True, **kwargs):
        if sidebyside:
            fig, ax = plt.subplots(1, 2, **kwargs)
            ax[0].imshow(x[:3].permute(1, 2, 0).numpy())
            ax[1].imshow(x[3].numpy(), cmap="gray")
        else:
            fig, ax = plt.subplots(1, 1, **kwargs)
            ax.imshow(x[:3].permute(1, 2, 0).numpy(), alpha=0.5)
            ax.imshow(x[3].numpy(), cmap="gray", alpha=0.5)
        return ax
