# Repository of datasets used for DS experiments

1. CMAPSS datasets was originally downloaded from [`data.nasa.gov`](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) but has since been redacted. For our purposes, it is now hosted in this repository.

## Create Environment
Set up a working miniconda environment as follows:
``` bash
conda create -n datasets python=3.10
conda activate datasets
pip install -r requirements.txt
pre-commit install
```
