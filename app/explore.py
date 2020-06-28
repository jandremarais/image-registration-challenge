import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

data = Path("/home/jan/data/aero")

train_master = pd.read_pickle(data / "master_manifest_train_red.pkl")
val_master = pd.read_pickle(data / "master_manifest_val_red.pkl")

train_ids = train_master["survey_id"].unique()
valid_ids = train_master["survey_id"].unique()


def show(
    im: Image.Image, ax: Optional[plt.Axes] = None, a: Optional[float] = None, **kwargs
) -> plt.Axes:
    im_array = np.array(im)
    if not ax:
        _, ax = plt.subplots(1, 1, **kwargs)
    nc = count_channels(im_array)
    if nc == 1:
        cmap = "gray"
    else:
        cmap = None
    ax.imshow(np.array(im), cmap=cmap, alpha=a)
    return ax


def load_pair(
    idx: int, path: Path, resize: bool = True
) -> np.ndarray:
    rgb = Image.open(path / f"survey_{idx}/reference/orthomosaic_visible.tif")
    red = Image.open(path / f"survey_{idx}/target/aligned/red/red.tif")
    # red = Image.open(path / f"survey_{idx}/target/misaligned/red/red-random.tif")
    if resize:
        rgb = rgb.resize(size=red.size, resample=Image.BILINEAR)
    a1 = np.array(rgb)
    a2 = np.array(red)

    assert a1.ndim == 3
    assert a2.ndim == 2
    assert a1.shape[-1] == 4

    return np.concatenate([a1, a2[..., None]], -1)


def show_pair(pair: np.ndarray, **kwargs):
    _, ax = plt.subplots(1, 2, **kwargs)
    ax[0].imshow(pair[..., :4].astype(np.int))
    ax[1].imshow(pair[..., 4:].squeeze(), cmap="gray")
    ax[0].grid()
    ax[1].grid()
    ax[0].set_title('RGB')
    ax[1].set_title('RED')
    return ax


def random_crop(pair: np.ndarray, size: int, threshold: float = 0.0) -> np.ndarray:
    h, w = pair.shape[:2]
    x0 = random.randint(0, w - size - 1)
    y0 = random.randint(0, h - size - 1)
    crop = pair[y0:(y0 + size), x0:(x0 + size)]
    return crop


x = []
y = []
for idx in train_ids[:3]:
    pair = load_pair(idx, data)

    nonnull = pair[..., 3] != 0
    x.append(pair[nonnull][..., :3])
    y.append(pair[nonnull][..., 4])


x = np.concatenate(x)
y = np.concatenate(y)

pair_val = load_pair(valid_ids[2], data)

nonnull = pair_val[..., 3] != 0
x_val = pair_val[nonnull][..., :3]
y_val = pair_val[nonnull][..., 4]


x_scaled = (x - np.mean(x, 0))/np.std(x, 0)
x_val_scaled = (x_val - np.mean(x, 0))/np.std(x, 0)

reg = linear_model.LinearRegression()

reg.fit(x_scaled, y)

reg.score(x_scaled, y)
reg.score(x_val_scaled, y_val)

# reg.coef_
y_pred = reg.predict(x_val_scaled)

rf = RandomForestRegressor(n_jobs=-1, max_samples=0.5, n_estimators=300)
rf.fit(x, y)
rf.score(x, y)
rf.score(x_val, y_val)
y_pred = rf.predict(x_val)


tmp = np.zeros_like(pair_val[..., 0])
tmp[nonnull] = y_pred

fig, ax = plt.subplots(1, 2, figsize=(20,20))
ax[0].imshow(tmp)
ax[1].imshow(pair_val[..., 4])

pair[..., 4].min()
tmp.min()
tmp.max()

np.mean((y_pred - y_val) ** 2)

pair[..., 4]
pair[]


for i in range(3):
    print(np.corrcoef(pair[:, :, i].reshape(-1), pair[:, :, 4].reshape(-1)))

np.corrcoef()
show_pair(pair[2000:2500, 400:900], figsize=(20, 12))

pair = load_pair(train_ids[5], data)
show_pair(pair[2000:2500, :500], figsize=(20, 12))


tmp =  random_crop(pair, 200)
show_pair(pair)


(tmp[..., 4] == 0).mean()


red_mal = Image.open(data / "survey_8502/target/misaligned/red/red-random.tif")
rgb_small = rgb.resize((red_al.size))


def count_channels(im: np.ndarray) -> int:
    if im.ndim == 3:
        return im.shape[-1]
    elif im.ndim == 2:
        return 1
    else:
        raise NotImplementedError


def overlay(im1: Image.Image, im2: Image.Image, **kwargs):
    ax = show(im1, a=0.5, **kwargs)
    ax = show(im2, ax=ax, a=0.5)
    return ax


# show(red_al)

overlay(rgb, red_al, figsize=(20, 20))
overlay(rgb_small, red_mal, figsize=(20, 20))
