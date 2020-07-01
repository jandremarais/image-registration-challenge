from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms


class Misaligned(Dataset):
    def __init__(
        self,
        path: Path,
        sz: int = 128,
        # mean=(0.4571, 0.4437, 0.3610, 0.0208),
        # std=(0.1910, 0.1690, 0.1431, 0.0111),
        mean=(0.4382, 0.0208),
        std=(0.1704, 0.0111)
    ):
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
                transforms.Grayscale(),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean[:3], self.std[:3]),
                transforms.Normalize(self.mean[0], self.std[0])
            ]
        )

        red_tfms = transforms.Compose(
            [
                transforms.ToPILImage(mode="F"),
                transforms.Resize(self.sz),
                transforms.ToTensor(),
                # transforms.Normalize(self.mean[3], self.std[3]),
                transforms.Normalize(self.mean[1], self.std[1])
            ]
        )
        return torch.cat([rgb_tfms(rgb), red_tfms(red)])

    def __getitem__(self, index):
        x = np.load(self.image_fns[index])
        y = np.load(self.target_fns[index])
        y *= self.sz
        x = self.transform(x)
        return (
            x,
            torch.tensor(
                [
                    [[y[0, 0], y[1, 0]], [y[3, 0], y[2, 0]]],
                    [[y[0, 1], y[1, 1]], [y[3, 1], y[2, 1]]],
                ]
            ),
        )

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
