# !%reload_ext autoreload
# !%autoreload 2
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
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
    dl = DataLoader(ds, num_workers=6, shuffle=False, batch_size=10)
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
for n, x in enumerate(iter(train_dl)):
    print(n)
    crop_list = tfms(x)
    fp = [data / f"crops/train/tile_{total + i}.npy" for i in range(len(x))]
    total += len(x)

    with ProcessPoolExecutor(max_workers=10) as e:
        e.map(_save, [(fp, a) for fp, a in zip(fp, crop_list)])

total = 0
for n, x in enumerate(iter(valid_dl)):
    print(n)
    crop_list = tfms(x)
    fp = [data / f"crops/valid/tile_{total + i}.npy" for i in range(len(x))]
    total += len(x)

    with ProcessPoolExecutor(max_workers=10) as e:
        e.map(_save, [(fp, a) for fp, a in zip(fp, crop_list)])
