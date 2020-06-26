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