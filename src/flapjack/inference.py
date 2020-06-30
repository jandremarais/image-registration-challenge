from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from flapjack.model import Model
from flapjack.utils import plot, warp_from_target


def predict_pair(rgb: np.ndarray, red: np.ndarray, model):
    """[summary]

    Args:
        rgb (np.ndarray): Image array or RGB tile with values in (0,255).
        red (np.ndarray): Image array of narrowbind tile with shape (H, W).
    """
    h, w = red.shape
    assert rgb.shape[:2] == (h, w)
    assert h == w
    x = np.concatenate([rgb.astype(np.float32), red[..., None]], -1)
    tx = model.valid_ds.transform(x)
    yhat = model(tx[None].to(model.device))
    yhat = yhat.view(4, 2).detach().cpu().numpy()
    return warp_from_target(x, yhat)


def predict_aero_pair(survey_id: int, tile: int, path: Path, model, crop: int = 1000):
    rgbp = path / f"survey_{survey_id}/reference/tile__{tile}.npy"
    redp = path / f"survey_{survey_id}/target/misaligned/red/tile__{tile}.npy"

    rgb = np.load(rgbp)[0][..., :3]
    red = np.load(redp)[0]

    rgb = cv2.resize(rgb, (red.shape[1], red.shape[0]))
    rgb = rgb[:crop, :crop]
    red = red[:crop, :crop]
    return predict_pair(rgb, red, model)


def predict_ortho(rgb, red):
    pass


model = Model.load_from_checkpoint(
    "lightning_logs/version_52/checkpoints/epoch=30.ckpt"
)

model.eval()

model.valid_ds.image_fns

data = Path('/home/jan/data/aero/')

px, mat = predict_aero_pair(11263, 6, data, model)

plot(px, figsize=(10, 10))
plot(px, figsize=(10, 10), sidebyside=False)
