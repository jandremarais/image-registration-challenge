import pytorch_lightning as pl
from flapjack.model import Model


model = Model.load_from_checkpoint(
    "lightning_logs/version_52/checkpoints/epoch=30.ckpt"
)

model.eval()

