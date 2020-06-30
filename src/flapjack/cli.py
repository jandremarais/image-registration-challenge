import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import typer

from flapjack.model import Model
from flapjack.utils import crop_misaligned, load_red, load_rgb

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


# def train

@app.command
def predict(
    rgb_path: Path,
    red_path: Path,
    ckpt: Path = Path("checkpoints/bst.ckpt"),
    dst: Path = Path("."),
):
    model = Model.load_from_checkpoint(ckpt)
    print(model.sz)


if __name__ == "__main__":
    app()
