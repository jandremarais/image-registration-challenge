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