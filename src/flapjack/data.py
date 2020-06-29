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


class Surveys(Dataset):
    def __init__(
        self,
        rgbs: List[Image.Image],
        reds: List[Image.Image],
        sz: int = 128,
        big_crop_sz: int = 300,
        crops_per_image: int = 50,
    ):
        """Initialise the dataset.

        Args:
            rgb_paths (List[Path]): Paths to the rgb orthomosaic tif files.
            red_paths (List[Path]): Paths to the RED (aligned) orthomosaic tif files
                in the same order as the RGB paths, such that the rgb_paths[i] and
                red_paths[i] is of the same field/survey.
            sz (int): final image side length.
            big_crop_sz (int): Side length of square crop to be taken on both images.
                The idea with this size reduction is to reduce computation time.
                But still keeping the image context to avoid empty edges.
            crops_per_image (int): Number of crops per survey.
        """
        assert len(rgbs) == len(reds)
        self.rgbs = [
            im1.resize(im2.size, resample=Image.BILINEAR)
            for im1, im2 in zip(rgbs, reds)
        ]
        self.reds = reds
        self.sz = sz
        self.big_crop_sz = big_crop_sz
        self.crops_per_image = crops_per_image

    def transform(
        self, rgb: Image.Image, red: Image.Image, a: float, dx: int, dy: int
    ) -> torch.Tensor:
        def _get_big_crop():
            bi, bj, bh, bw = transforms.RandomCrop.get_params(
                red, (self.big_crop_sz, self.big_crop_sz)
            )

            big_crop = partial(TF.crop, top=bi, left=bj, height=bh, width=bw)

            tfms = transforms.Compose([big_crop, transforms.ToTensor()])
            tmp = tfms(rgb)
            alpha_c = tmp[3]
            if (alpha_c > 0).float().mean() < 0.5:
                return _get_big_crop()
            else:
                return big_crop

        big_crop = _get_big_crop()
        a1 = transforms.RandomRotation.get_params((-360, 360))
        rotate1 = partial(TF.rotate, angle=a1, resample=Image.BILINEAR)

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

        center_crop = transforms.CenterCrop(self.sz)

        rgb_tfms = transforms.Compose(
            [big_crop, vflip, hflip, rotate1, center_crop, transforms.ToTensor()]
        )
        affine_tfms = partial(
            TF.affine,
            angle=a,
            translate=(dx, dy),
            scale=1,
            shear=0,
            resample=Image.BILINEAR,
        )
        red_tfms = transforms.Compose(
            [
                big_crop,
                vflip,
                hflip,
                rotate1,
                affine_tfms,
                center_crop,
                transforms.ToTensor(),
            ]
        )
        return torch.cat([rgb_tfms(rgb)[:3], red_tfms(red)])

    def __getitem__(self, index):
        i = index // self.crops_per_image
        rgb = self.rgbs[i]
        red = self.reds[i]
        a, (dx, dy), _, _ = transforms.RandomAffine.get_params(
            (-15, 15), (0.1, 0.1), (1, 1), (0, 0), (self.sz, self.sz),
        )
        x = self.transform(rgb, red, a, dx, dy)
        return x, torch.tensor([a, dx, dy], dtype=torch.float32)

    def __len__(self):
        return len(self.rgbs) * self.crops_per_image

    @staticmethod
    def plot_pair(t: torch.Tensor, **kwargs):
        rgb = t[:3]
        red = t[-1]
        fig, ax = plt.subplots(1, 2, **kwargs)
        ax[0].imshow(rgb.permute(1, 2, 0))
        ax[1].imshow(red, cmap="gray")
        ax[0].grid()
        ax[1].grid()
        return ax

    @staticmethod
    def overlay(t: torch.Tensor, **kwargs):
        print(t.shape)
        rgb = t[:3]
        red = t[-1]
        plt.figure(**kwargs)
        plt.imshow(rgb.permute(1, 2, 0))
        plt.imshow(red, alpha=0.5)

    def show(self, index, **kwargs):
        t, (a, dx, dy) = self[index]
        ax = self.plot_pair(t, **kwargs)
        ax[1].set_title(f"angle: {a.round()}, dx: {dx}, dy: {dy}")
        return ax


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
        self, path: Path, sz: int = 128, angle: int = 15, translation: float = 0.1
    ):
        super().__init__()
        self.filenames = list(path.iterdir())
        self.sz = sz
        self.angle = angle
        self.translation = translation
        self.mean = (0, 0, 0, 0)
        self.std = (1, 1, 1, 1)

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
                transforms.ToPILImage(mode='F'),
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
            ax[1].imshow(x[3].numpy(), cmap='gray')
        else:
            fig, ax = plt.subplots(1, 1, **kwargs)
            ax.imshow(x[:3].permute(1, 2, 0).numpy(), alpha=0.5)
            ax.imshow(x[3].numpy(), cmap='gray', alpha=0.5)
        return ax
