# !%reload_ext autoreload
# !%autoreload 2
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from flapjack.model import Model
from flapjack.utils import plot, warp_from_target


def transform(x, sz=224, mean=(0.4382, 0.0208), std=(0.1704, 0.0111)):
    rgb = x[..., :3].astype(np.uint8)
    red = x[..., 3]

    rgb_tfms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(sz),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean[0], std[0]),
        ]
    )

    red_tfms = transforms.Compose(
        [
            transforms.ToPILImage(mode="F"),
            transforms.Resize(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean[1], std[1]),
        ]
    )
    return torch.cat([rgb_tfms(rgb), red_tfms(red)])


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
    tx = transform(x)
    yhat = model(tx[None].to(model.device))
    yhat = yhat.view(4, 2).detach().cpu().numpy()
    return warp_from_target(x, yhat)


def predict_aero_pair(survey_id: int, tile: int, path: Path, model):
    rgbp = path / f"survey_{survey_id}/reference/tile__{tile}.npy"
    redp = path / f"survey_{survey_id}/target/misaligned/red/tile__{tile}.npy"

    rgb = np.load(rgbp)[0][..., :3]
    red = np.load(redp)[0]
    min_sz = min(red.shape)
    print(min_sz)

    rgb = cv2.resize(rgb, (red.shape[1], red.shape[0]))
    rgb = rgb[:min_sz, :min_sz]
    red = red[:min_sz, :min_sz]
    return predict_pair(rgb, red, model)


def predict_ortho(rgb, red):
    pass


model = Model.load_from_checkpoint(
    "lightning_logs/version_56/checkpoints/epoch=30.ckpt"
)

model.eval()

data = Path('/home/jan/data/aero/')

px, mat = predict_aero_pair(8493, 3, data, model)
plot(px, figsize=(10, 10), sidebyside=False)

plot(px[800:900, 0:200], figsize=(20, 10))


from flapjack.data import Misaligned

ds = Misaligned(data/'crops/valid', sz=224)

x, y = ds[0]

yhat = model(x[None].to(model.device))
yhat = yhat.view(4, 2).detach().cpu().numpy()
warp_from_target(x, yhat)

x[..., -1].shape

yhat
y.view(4, 2)