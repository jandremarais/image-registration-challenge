import random
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

# STEPS
# load rgb
# load red
# resize rgb to red size (since red is smaller)
# sample a random point in rgb
# get the corners of a square with this point as the center
# rotate these corners at a random angle
# crop the rgb image defined by these corners
# do a random translation of the center
# get the corners of a square with this point as the center
# rotate these corners at a random angle within margin or previous angle
# crop the red image defined by these corners


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
    sz: int = 400,
    max_angle: int = 10,
    max_shift: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = red.shape
    assert rgb.shape[:2] == (H, W)

    # consider sampling from mask
    cx, cy = (random.randint(0, W - sz / 2), random.randint(0, H - sz / 2))
    rgb_corners = square_from_center(cx, cy, sz)

    a = random.randint(0, 360)
    rgb_corners = rotate_points(rgb_corners, a, (cx, cy))

    dst = np.array([[0, 0], [sz - 1, 0], [sz - 1, sz - 1], [0, sz - 1]]).astype(
        np.float32
    )

    warp_mat = cv2.getAffineTransform(rgb_corners[:3], dst[:3])
    rgb_crop = cv2.warpAffine(rgb, warp_mat, (sz, sz))

    a = random.randint(-max_angle, max_angle)
    red_corners = rotate_points(rgb_corners, a, (cx, cy))

    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    red_corners[:, 0] += dx
    red_corners[:, 1] += dy

    warp_mat = cv2.getAffineTransform(red_corners[:3], dst[:3])
    red_crop = cv2.warpAffine(red, warp_mat, (sz, sz))

    rgb_corners2 = np.pad(rgb_corners.T, ((0, 1), (0, 0)), constant_values=1)
    rgb_corners2 = np.matmul(warp_mat, rgb_corners2).T.astype(np.float32)

    diff_corners = dst - rgb_corners2

    x = np.concatenate([rgb_crop.astype(np.float32), red_crop[..., None]], -1)

    return x, diff_corners


data = Path("/home/jan/data/aero")
master_train = pd.read_pickle(data / "master_manifest_train_red.pkl")
master_valid = pd.read_pickle(data / "master_manifest_val_red.pkl")

train_ids = master_train["survey_id"].unique()
valid_ids = master_valid["survey_id"].unique()

images_per_survey = 100
sz = 400
max_angle = 15
max_shift = 30

folder = "crops"

for split in ["train", "valid"]:
    for t in ["images", "targets"]:
        (data / f"{folder}/{split}/{t}").mkdir(parents=True, exist_ok=True)

for idx in train_ids:
    rgb = load_rgb(idx, data)
    red = load_red(idx, data)
    H, W = red.shape
    rgb = cv2.resize(rgb, (W, H), cv2.INTER_LINEAR)

    n = 0
    while n < images_per_survey:
        x, y = crop_misaligned(
            rgb, red, sz=sz, max_angle=max_angle, max_shift=max_shift
        )

        if (x[..., 0] == 0).mean() > 0.5:
            continue
        else:
            ts = int(time.time() * 1000)
            np.save(data / f"{folder}/train/images/image_{idx}_{ts}.npy", x)
            np.save(data / f"{folder}/train/targets/target_{idx}_{ts}.npy", y)
            n += 1


for idx in valid_ids:
    rgb = load_rgb(idx, data)
    red = load_red(idx, data)
    H, W = red.shape
    rgb = cv2.resize(rgb, (W, H), cv2.INTER_LINEAR)

    n = 0
    while n < images_per_survey:
        x, y = crop_misaligned(
            rgb, red, sz=sz, max_angle=max_angle, max_shift=max_shift
        )

        if (x[..., 0] == 0).mean() > 0.5:
            continue
        else:
            ts = int(time.time() * 1000)
            np.save(data / f"{folder}/valid/images/image_{idx}_{ts}.npy", x)
            np.save(data / f"{folder}/valid/targets/target_{idx}_{ts}.npy", y)
            n += 1


# DONT DELETE
# warp_mat = cv2.getAffineTransform(dst[:3] - diff_corners[:3], dst[:3])
# red_aligned_crop = cv2.warpAffine(red_crop, warp_mat, (sz, sz))
