import random
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class Tiles(Dataset):
    def __init__(
        self,
        rgbs: List[Image.Image],
        reds: List[Image.Image],
        sz: int = 500,
        crops_per_image: int = 50,
    ):
        assert len(rgbs) == len(reds)
        self.rgbs = [
            im1.resize(im2.size, resample=Image.BILINEAR)
            for im1, im2 in zip(rgbs, reds)
        ]
        self.reds = reds
        self.sz = sz
        self.crops_per_image = crops_per_image

    def transform(self, rgb: Image.Image, red: Image.Image) -> torch.Tensor:

        a1 = transforms.RandomRotation.get_params((-360, 360))
        rotate1 = partial(TF.rotate, angle=a1, resample=Image.BILINEAR)

        bi, bj, bh, bw = transforms.RandomCrop.get_params(red, (self.sz, self.sz))

        big_crop = partial(TF.crop, top=bi, left=bj, height=bh, width=bw)

        def _noop(x):
            return x

        if random.random() > 0.5:
            hflip = TF.hflip
        else:
            hflip = _noop

        if random.random() > 0.5:
            vflip = TF.vflip
        else:
            vflip = _noop

        tfms = transforms.Compose(
            [rotate1, big_crop, vflip, hflip, transforms.ToTensor()]
        )

        trgb = tfms(rgb)
        complete = trgb[-1].float().mean()
        if complete < 0.5:
            return self.transform(rgb, red)
        else:
            return torch.cat([trgb[:3], tfms(red)])

    def __getitem__(self, index):
        i = index // self.crops_per_image
        rgb = self.rgbs[i]
        red = self.reds[i]
        x = self.transform(rgb, red)
        return x

    def __len__(self):
        return len(self.rgbs) * self.crops_per_image


class Misaligned(Dataset):
    def __init__(
        self,
        path: Path,
        sz: int = 128,
        angle: int = 15,
        translation: float = 0.1,
        mean: Tuple[float, float, float, float] = (0.5107, 0.4931, 0.4041, 0.0231),
        std: Tuple[float, float, float, float] = (0.1560, 0.1328, 0.1185, 0.0095),
    ):
        super().__init__()
        self.filenames = list(path.iterdir())
        self.sz = sz
        self.angle = angle
        self.translation = translation
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.filenames)

    def rgb_transform(self, x):
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.CenterCrop(self.sz),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[:3], self.std[:3]),
            ]
        )
        return tfms(x)

    def red_transform(self, x, angle, dx, dy):
        affine_tfms = partial(
            TF.affine,
            angle=angle,
            translate=(dx, dy),
            scale=1,
            shear=0,
            resample=Image.BILINEAR,
        )
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(mode="F"),
                affine_tfms,
                transforms.CenterCrop(self.sz),
                transforms.ToTensor(),
                transforms.Normalize(self.mean[3], self.std[3]),
            ]
        )
        return tfms(x)

    def transform(self, x, angle, dx, dy):
        return torch.cat(
            [
                self.rgb_transform(x[..., :3]),
                self.red_transform(x[..., 3], angle, dx, dy),
            ]
        )

    def sample_y(self):
        a, (dx, dy), _, _ = transforms.RandomAffine.get_params(
            (-self.angle, self.angle),
            (self.translation, self.translation),
            (1, 1),
            (0, 0),
            (self.sz, self.sz),
        )
        return a, dx, dy

    def __getitem__(self, index):
        x = np.load(self.filenames[index])
        y = self.sample_y()
        x = self.transform(x, *y)
        return x, torch.tensor(y, dtype=torch.float32)

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
