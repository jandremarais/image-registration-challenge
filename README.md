# image-registration-challenge
Code to align RGB and narrowband RED aerial imagery. This work is part of the Aerobotics Computer Vision Engineer project.

## Overview

This repo contains code to build a CNN that can align a RGB, RED narrowband pair of images.
The CNN accepts the misaligned pair as a stacked tensor and predicts the offsets between their corners.
From these offsets we can solve the affine transformation that will align the RED image with the RGB.

The main features of this project are:
- creating a synthetic dataset of misaligned pairs used for training.
- training and evaluating the CNN
- simple inference pipeline on user provided pairs

## Getting started

The simplest way to run this code on your machine is to do so in a docker container.
You will need to build the image with
```
docker build --rm -f "src/Dockerfile" -t imageregistrationchallenge:latest "src"
```

and then run the container with:

```
docker run --rm -it --gpus all imageregistrationchallenge
```
if you are only going to run predictions, the 'gpus' flag is not required.

If you don't care to install any missing system dependencies, feel free to just run:
```
python -m venv /path/to/venv
source /path/to/venv/bin/activate
cd src
poetry install
```
to create a virtual environment with all the python dependencies installed.


## Basic usage


### Using model from checkpoint

```
python src/flapjack/cli.py predict.py /path/to/rgb.npy /path/to/red.npy /path/to/checkpoint dst
```

### Training your own model

This will train the model with the same starting point as the one reported on:

```
python src/flapjack/train.py --epochs 20 --batch_size 64 --nw 8 --data_path /home/jan/data/aero/crops --learning_rate 0.0001 --gpus 1
```

This will also store the logs in `./lightning_logs`.
You can run `tensorboard --logdir ./lightning_logs` to inspect the loss curves as they train.

## To Do

- Write tests for main functions
- implement alignment based on multiple patches
- make CLI more user friendly and type safe.
- remove deprecated code
- implement hyperparemeter search
