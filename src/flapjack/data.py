import random
from functools import partial
from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class RgbRed(Dataset):
    def __init__(
        self,
        rgb_paths: List[Path],
        red_paths: List[Path],
        sz: int = 128,
        big_crop_sz: int = 300,
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
        """
        assert len(rgb_paths) == len(red_paths)
        self.rgb_paths = rgb_paths
        self.red_paths = red_paths
        self.sz = sz
        self.big_crop_sz = big_crop_sz

    def transform(
        self, rgb: Image.Image, red: Image.Image, a: float, dx: int, dy: int
    ) -> torch.Tensor:
        red_w, red_h = red.size

        bi, bj, bh, bw = transforms.RandomCrop.get_params(
            red, (self.big_crop_sz, self.big_crop_sz)
        )

        big_crop = partial(TF.crop, top=bi, left=bj, height=bh, width=bw)

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
            [
                transforms.Resize((red_h, red_w)),
                big_crop,
                vflip,
                hflip,
                rotate1,
                center_crop,
                transforms.ToTensor(),
            ]
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
        return torch.cat([rgb_tfms(rgb), red_tfms(red)])

    def __getitem__(self, index):
        rgb = Image.open(self.rgb_paths[index])
        red = Image.open(self.red_paths[index])
        a, (dx, dy), _, _ = transforms.RandomAffine.get_params(
            (-15, 15),
            (0.05, 0.05),
            (1, 1),
            (0, 0),
            (self.big_crop_sz, self.big_crop_sz),
        )
        x = self.transform(rgb, red, a, dx, dy)
        return x, (a, dx, dy)

    def __len__(self):
        return len(self.rgb_paths)

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

    def show(self, index, **kwargs):
        t, (a, dx, dy) = self[index]
        ax = self.plot_pair(t, **kwargs)
        ax[1].set_title(f"angle: {round(a, 2)}, dx: {dx}, dy: {dy}")
        return ax
