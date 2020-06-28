from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from flapjack.data import Tiles


def paths_from_master(df: pd.DataFrame):
    ids = df['survey_id'].unique()
    rgb_paths = [f"survey_{idx}/reference/orthomosaic_visible.tif" for idx in ids]
    red_paths = [f"survey_{idx}/target/aligned/red/red.tif" for idx in ids]
    return rgb_paths, red_paths


def load_from_paths(paths: List[Path], data: Path) -> List[Image.Image]:
    return [Image.open(data / o) for o in paths]


def dl_from_master(df: pd.DataFrame, data: Path):
    rgb_paths, red_paths = paths_from_master(df)
    rgb_ims = load_from_paths(rgb_paths, data)
    red_ims = load_from_paths(red_paths, data)

    ds = Tiles(rgb_ims, red_ims, crops_per_image=100)
    dl = DataLoader(ds, num_workers=1, shuffle=False, batch_size=16)
    return dl


data = Path('/home/jan/data/aero')
master_train = pd.read_pickle(data / "master_manifest_train_red.pkl")
master_valid = pd.read_pickle(data / "master_manifest_val_red.pkl")

train_dl = dl_from_master(master_train, data)
valid_dl = dl_from_master(master_valid, data)

tfms = transforms.Compose([
    transforms.Lambda(lambda x: torch.unbind(x)),
    transforms.Lambda(lambda x: [xx.permute(1, 2, 0).numpy() for xx in x])
])


def _save(o):
    np.save(o[0], o[1])


total = 0
for n, (x1, x2) in enumerate(iter(train_dl)):
    print(n)
    crop_list_rgb = tfms(x1)
    crop_list_red = tfms(x2)
    rgb_fp = [data / f"crops/train/rgb/tile_rgb_{total + i}.npy" for i in range(len(x1))]
    red_fp = [data / f"crops/train/red/tile_red_{total + i}.npy" for i in range(len(x1))]
    total += n

    with ProcessPoolExecutor(max_workers=12) as e:
        e.map(_save, [(fp, a) for fp, a in zip(rgb_fp, crop_list_rgb)])

    with ProcessPoolExecutor(max_workers=12) as e:
        e.map(_save, [(fp, a) for fp, a in zip(red_fp, crop_list_red)])

total = 0
for n, (x1, x2) in enumerate(iter(valid_dl)):

    crop_list_rgb = tfms(x1)
    crop_list_red = tfms(x2)
    rgb_fp = [data / f"crops/valid/rgb/tile_rgb_{total + i}.npy" for i in range(len(x1))]
    red_fp = [data / f"crops/valid/red/tile_red_{total + i}.npy" for i in range(len(x1))]
    total += n

    with ProcessPoolExecutor(max_workers=12) as e:
        e.map(_save, [(fp, a) for fp, a in zip(rgb_fp, crop_list_rgb)])

    with ProcessPoolExecutor(max_workers=12) as e:
        e.map(_save, [(fp, a) for fp, a in zip(red_fp, crop_list_red)])


