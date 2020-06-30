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
python src/predict.py /path/to/rgb.npy /path/to/red.npy /path/to/checkpoint dst
```

### Training your own model

## Approach

We attempt to solve the multimodal image registration problem by training a CNN to estimate the mapping between a pair of images by predicting the 4-point homgraphy given the pair of images stacked channel-wise.

The first step is to create a dataset for a CNN to learn from.
This process to create one training example is as follows:

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