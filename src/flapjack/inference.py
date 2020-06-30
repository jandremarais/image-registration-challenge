from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from flapjack.model import Model
from flapjack.utils import warp_from_target, plot

model = Model.load_from_checkpoint(
    "lightning_logs/version_52/checkpoints/epoch=30.ckpt"
)

model.eval()

model.valid_ds.image_fns

# predict on pairs
# predict on tile
# predict on ortho



data = Path('/home/jan/data/aero/')

rgbp = data / 'crops/valid/images/image_8494_1593477617381.npy'
redp = data / 'crops/valid/targets/target_8494_1593477617381.npy'

im_all = np.load(rgbp)
rgb = im_all[..., :3]
red = im_all[..., 3]

tfms = transforms.Compose([

])





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


px, mat = predict_pair(rgb, red, model)

mat

plot(px, figsize=(10, 10))
plot(px, figsize=(10, 10), sidebyside=False)


torch.flatten(torch.tensor([[1, 2], [3, 4]]))

rgb.max()
