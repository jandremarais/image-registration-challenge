import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import pandas as pd
from pathlib import Path
from .data import Surveys
from torch.utils.data import DataLoader
from argparse import ArgumentParser


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.l1 = nn.Linear(128 * 128 * 4, 3)

    def forward(self, x):
        return self.l1(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        logs = {"loss": F.mse_loss(y_hat, y)}
        return {"loss": F.mse_loss(y_hat, y), "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {"val_loss": F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss_mean": val_loss_mean}
        return {"val_loss": val_loss_mean, "log": logs}

    def prepare_data(self):
        data = Path(self.hparams.data_path)

        master_train = pd.read_pickle(data / "master_manifest_train_red.pkl")
        master_valid = pd.read_pickle(data / "master_manifest_val_red.pkl")
        train_ids = master_train["survey_id"].unique()
        valid_ids = master_valid["survey_id"].unique()

        train_rgb_paths = [
            data / f"survey_{idx}/reference/orthomosaic_visible.tif"
            for idx in train_ids
        ]
        train_red_paths = [
            data / f"survey_{idx}/target/aligned/red/red.tif" for idx in train_ids
        ]
        valid_rgb_paths = [
            data / f"survey_{idx}/reference/orthomosaic_visible.tif"
            for idx in valid_ids
        ]
        valid_red_paths = [
            data / f"survey_{idx}/target/aligned/red/red.tif" for idx in valid_ids
        ]

        train_rgbs = [Image.open(o) for o in train_rgb_paths]
        train_reds = [Image.open(o) for o in train_red_paths]
        valid_rgbs = [Image.open(o) for o in valid_rgb_paths]
        valid_reds = [Image.open(o) for o in valid_red_paths]

        self.train_ds = Surveys(train_rgbs, train_reds, sz=128, big_crop_sz=500)
        self.valid_ds = Surveys(valid_rgbs, valid_reds, sz=128, big_crop_sz=500)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.batch_size * 2, shuffle=False
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=0.02, type=float)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--data_path", type=str, default="./")
        return parser
