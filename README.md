# image-registration-challenge
Code to align RGB and narrowband RED aerial imagery. This work is part of the Aerobotics Computer Vision Engineer project.

## Repo Overview

The `app` directory contains the main scripts to complete the challenge.
It also has a `pyproject.toml` file that specifies the python dependencies required to run the scripts.

Inside `src` we have the core functions and utilities to perform image registration wrapped as a python package that is installable with:

```
cd src
poetry install 
```

## Getting started

The simplest way to run this code on your machine is to do it in a docker container.
You will need to build the image with
```
docker build ...
```

or you can pull it form github actions with:

```
docker pull ...
```

If you don't care to install any missing system dependencies, feel free to just run:
```
python -m venv /path/to/venv
source /path/to/venv/bin/activate
cd app
poetry install
```
to create a virtual environment with all the python dependencies installed.

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