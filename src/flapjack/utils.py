from typing import Tuple

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


def compute_stats(ds: Dataset):
    dl = DataLoader(
        ds,
        batch_size=32,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for x, y in dl:
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


def warp_from_target(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sz = x.shape[0]
    dst = np.array([[0, 0], [sz - 1, 0], [sz - 1, sz - 1], [0, sz - 1]]).astype(
        np.float32
    )
    warp_mat = cv2.getAffineTransform(dst[:3] - y[:3], dst[:3])
    red_aligned = cv2.warpAffine(x[..., -1], warp_mat, (sz, sz))
    return np.concatenate([x[..., :3], red_aligned[..., None]], -1), warp_mat


def plot(x, y=None, sidebyside=True, **kwargs):
    if sidebyside:
        fig, ax = plt.subplots(1, 2, **kwargs)
        ax[0].imshow(x[..., :3].astype(np.uint8))
        ax[1].imshow(x[..., 3], cmap="gray")
    else:
        fig, ax = plt.subplots(1, 1, **kwargs)
        ax.imshow(x[..., :3].astype(np.uint8), alpha=0.5)
        ax.imshow(x[..., 3], cmap="gray", alpha=0.5)
    return ax