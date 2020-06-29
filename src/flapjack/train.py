import pytorch_lightning as pl
from flapjack.model import Model
from argparse import ArgumentParser


parser = ArgumentParser()

parser = Model.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

trainer = pl.Trainer.from_argparse_args(args)

model = Model(args)
trainer.fit(model)
