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

1. Load a corresponding pair of RGB and RED (aligned) images and ensure they are of the same width and height.
2. Apply a random translation and rotation to the RED image and record the transform parameters.
3. Perform a random square crop of both images at the same pixel locations, where the size is also randomly sampled from an appropriate range. If the crop has too much empty space, reject it.
4. Compute the horizontal and vertical pixel distances between the corners of the RED image and their corresponding locations in the transformed images.
5. Concatenate the RGB image with the RED image channel-wise and save it as a .tif file.
6. Save the 8 values defining the corner deltas as a .npy file.