import numpy as np
from pathlib import Path
import cv2
from matplotlib import pyplot as plt


def load_rgb(survey_id: int, tile: int, path: Path) -> np.ndarray:
    fn = path / f"survey_{survey_id}/reference/tile__{tile}.npy"
    rgb = np.load(str(fn))[0][..., :3]
    return rgb


def load_red(survey_id: int, tile: int, path: Path, aligned: bool = True) -> np.ndarray:
    if aligned:
        d = "aligned"
    else:
        d = "misaligned"
    fn = path / f"survey_{survey_id}/target/{d}/red/tile__{tile}.npy"
    red = np.load(str(fn))[0]
    return red


def resize_to_red(rgb, red):
    h, w = red.shape
    rgb = cv2.resize(rgb, (w, h), cv2.INTER_LINEAR)
    return rgb


def plot_pair(
    rgb: np.ndarray, red: np.ndarray, sidebyside: bool = True, ax=None, **kwargs
):
    if sidebyside:
        if not ax:
            fig, ax = plt.subplots(1, 2, **kwargs)
        ax[0].imshow(rgb)
        ax[1].imshow(red)
    else:
        if not ax:
            fig, ax = plt.subplots(1, 1, **kwargs)
        ax.imshow(rgb, alpha=0.5)
        ax.imshow(red, alpha=0.5)

    return ax


data = Path("/home/jan/data/aero")
list(data.iterdir())
s = 8494
tile = 9
rgb = load_rgb(s, tile, data)
red = load_red(s, tile, data, True)

rgb = resize_to_red(rgb, red)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

plot_pair(rgb, red, True, figsize=(20,20))
plot_pair(gray, red, True, figsize=(20,20))

sift = cv2.xfeatures2d.SIFT_create()



