import pytorch_lightning as pl
from .model import Model
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

trainer = pl.Trainer(gpus=1, max_epochs=3)
model = Model()
trainer.fit(model)
