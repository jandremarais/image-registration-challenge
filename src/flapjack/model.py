import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from .data import Misaligned
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

        self.train_ds = Misaligned(data / "train")
        self.valid_ds = Misaligned(data / "valid")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.nw,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.nw,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=0.02, type=float)
        parser.add_argument("--batch_size", default=8, type=int)
        parser.add_argument("--nw", default=1, type=int)
        parser.add_argument("--data_path", type=str, default="./")
        return parser
