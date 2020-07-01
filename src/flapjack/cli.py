import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import typer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from flapjack.data import Misaligned
from flapjack.inference import predict_pair
from flapjack.model import Model
from flapjack.utils import crop_misaligned, load_red, load_rgb, resize_to

app = typer.Typer()


@app.command()
def make_dataset(
    data: Path,
    img_folder: str = "crops",
    images_per_survey: int = 100,
    sz_range: Tuple[int, int] = (300, 600),
    max_angle: int = 15,
    max_shift: float = 0.1,
):
    master_train = pd.read_pickle(data / "master_manifest_train_red.pkl")
    master_valid = pd.read_pickle(data / "master_manifest_val_red.pkl")

    train_ids = master_train["survey_id"].unique()
    valid_ids = master_valid["survey_id"].unique()

    for split in ["train", "valid"]:
        for t in ["images", "targets"]:
            (data / f"{img_folder}/{split}/{t}").mkdir(parents=True, exist_ok=True)

    def _inner(ids, dstype):
        for idx in ids:
            typer.echo(f"Processing survey {idx} in {dstype}")
            rgb = load_rgb(idx, data)
            red = load_red(idx, data)
            H, W = red.shape
            rgb = cv2.resize(rgb, (W, H), cv2.INTER_LINEAR)

            n = 0
            while n < images_per_survey:
                x, y = crop_misaligned(
                    rgb,
                    red,
                    sz_range=sz_range,
                    max_angle=max_angle,
                    max_shift=max_shift,
                    normalize=True,
                )

                if (x[..., 0] == 0).mean() > 0.5:
                    continue
                else:
                    ts = int(time.time() * 1000)
                    np.save(
                        data / f"{img_folder}/{dstype}/images/image_{idx}_{ts}.npy", x
                    )
                    np.save(
                        data / f"{img_folder}/{dstype}/targets/target_{idx}_{ts}.npy", y
                    )
                    n += 1

    for ids, dstype in zip([train_ids, valid_ids], ["train", "valid"]):
        _inner(ids, dstype)

    typer.echo("Completed!")


@app.command()
def predict_tile_pair(
    rgb_path: str,
    red_path: str,
    ckpt: str = "checkpoints/bst.ckpt"
):
    model = Model.load_from_checkpoint(ckpt)

    rgb = np.load(rgb_path).squeeze()[..., :3]
    red = np.load(red_path).squeeze()

    rgb = resize_to(rgb, red)
    # sz = min(red.shape)
    sz=400

    red = red[:sz, :sz]
    rgb = rgb[:sz, :sz]

    _, mat = predict_pair(rgb, red, model)
    red_a = cv2.warpAffine(red, mat, (sz, sz))
    print(mat)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    ax[0].imshow(rgb)
    ax[1].imshow(red)
    ax[2].imshow(red_a)
    ax[0].set_title('RGB')
    ax[1].set_title('Misaligned RED')
    ax[2].set_title('Aligned RED')
    for a in ax: a.grid()
    plt.show()


@app.command()
def evaluate(valid_path: str, ckpt: str):
    model = Model.load_from_checkpoint(ckpt)
    ds = Misaligned(Path(valid_path))
    dl = DataLoader(ds, num_workers=6, shuffle=False, batch_size=16)
    model.eval()

    total_dist = 0
    n = 0
    with torch.no_grad():
        for x, y in dl:
            yhat = model(x)
            cdist = ((y - yhat) ** 2).sum(-1).sqrt()
            total_dist += cdist.sum()
            n += len(x)

        mace = total_dist/n
    typer.echo(f'MACE: {mace}')


if __name__ == "__main__":
    app()
