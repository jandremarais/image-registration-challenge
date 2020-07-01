from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from flapjack.utils import warp_from_target


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
    yhat = yhat[0].detach().cpu().numpy()
    pts = np.array([yhat[:, 0, 0], yhat[:, 0, 1], yhat[:, 1, 1], yhat[:, 1, 0]])
    return warp_from_target(x, pts)


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
