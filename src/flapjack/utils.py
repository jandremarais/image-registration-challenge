import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset


def square_from_center(cx, cy, sz):
    return np.array(
        [
            [cx - sz / 2, cy - sz / 2],
            [cx + sz / 2, cy - sz / 2],
            [cx + sz / 2, cy + sz / 2],
            [cx - sz / 2, cy + sz / 2],
        ]
    )


def load_rgb(idx: int, path: Path) -> np.ndarray:
    rgb_path = path / f"survey_{idx}/reference/orthomosaic_visible.tif"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
    # msk = rgb[..., -1]
    rgb = rgb[..., :3]
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


def load_red(idx: int, path: Path) -> np.ndarray:
    red_path = path / f"survey_{idx}/target/aligned/red/red.tif"
    red = cv2.imread(str(red_path), cv2.IMREAD_UNCHANGED)
    return red


def rotate_points(
    points: np.ndarray, angle: int, center: Tuple[float, float]
) -> np.ndarray:
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1)
    assert points.shape[1] == 2
    assert points.ndim == 2
    rpoints = np.pad(points.T, ((0, 1), (0, 0)), constant_values=1)
    rpoints = np.matmul(rot_mat, rpoints).T.astype(np.float32)
    return rpoints


def crop_misaligned(
    rgb: np.ndarray,
    red: np.ndarray,
    sz_range: Tuple[int, int] = (300, 600),
    max_angle: int = 10,
    max_shift: float = 0.1,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = red.shape
    assert rgb.shape[:2] == (H, W)

    sz = random.randint(*sz_range)
    # consider sampling from mask
    cx, cy = (random.randint(0, W - sz // 2), random.randint(0, H - sz // 2))
    rgb_corners = square_from_center(cx, cy, sz)

    a = random.randint(0, 360)
    rgb_corners = rotate_points(rgb_corners, a, (cx, cy))

    dst = np.array([[0, 0], [sz, 0], [sz, sz], [0, sz]]).astype(
        np.float32
    )

    warp_mat = cv2.getAffineTransform(rgb_corners[:3], dst[:3])
    rgb_crop = cv2.warpAffine(rgb, warp_mat, (sz, sz))

    a = random.randint(-max_angle, max_angle)
    red_corners = rotate_points(rgb_corners, a, (cx, cy))

    dx = random.randint(int(-max_shift * sz), int(max_shift * sz))
    dy = random.randint(int(-max_shift * sz), int(max_shift * sz))

    red_corners[:, 0] += dx
    red_corners[:, 1] += dy

    warp_mat = cv2.getAffineTransform(red_corners[:3], dst[:3])
    red_crop = cv2.warpAffine(red, warp_mat, (sz, sz))

    rgb_corners2 = np.pad(rgb_corners.T, ((0, 1), (0, 0)), constant_values=1)
    rgb_corners2 = np.matmul(warp_mat, rgb_corners2).T.astype(np.float32)

    diff_corners = dst - rgb_corners2

    x = np.concatenate([rgb_crop.astype(np.float32), red_crop[..., None]], -1)

    if normalize:
        diff_corners /= sz
    return x, diff_corners


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


def warp_from_target(red: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sz = red.shape[0]
    dst = np.array([[0, 0], [sz - 1, 0], [sz - 1, sz - 1], [0, sz - 1]]).astype(
        np.float32
    )
    warp_mat = cv2.getAffineTransform(dst[:3] - y[:3], dst[:3])
    red_aligned = cv2.warpAffine(red, warp_mat, (sz, sz))
    return red_aligned, warp_mat


def resize_to(im, ref):
    h, w = ref.shape[:2]
    return cv2.resize(im, (w, h), cv2.INTER_LINEAR)


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


def check_random_synth_sample(path: Path):
    """Visualise synthetic data sample.
    Plots RGB, RED and aligned RED (based on y).

    Args:
        path (Path): Path to synthetic misaligned crops.
    """
    images = list((path / "images").iterdir())

    img_fn = random.choice(images)
    tgt_fn = img_fn.name.replace("image", "target")
    tgt_fn = path / f"targets/{tgt_fn}"

    x = np.load(img_fn)
    y = np.load(tgt_fn)

    rgb = x[..., :3].astype("uint8")
    red = x[..., 3]

    sz = x.shape[0]
    dst = np.array([[0, 0], [sz, 0], [sz, sz], [0, sz]]).astype(np.float32)
    src = dst - y * sz

    mat = cv2.getAffineTransform(src[:3], dst[:3])
    red_a = cv2.warpAffine(red, mat, (sz, sz))

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(rgb)
    ax[1].imshow(red)
    ax[2].imshow(red_a)

    for a in ax: a.grid(color='w')
