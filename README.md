# Repository of datasets used for DS experiments

## Usage
1. The purpose of this repository is to host T-DAB datasets and any accompanying scripts used to prepare the data for an ML use case.
2. It should not contain any experimentation notebooks. For this, a separate repo ML-experiments have been created.

## Create Environment
Set up a working miniconda environment as follows:
``` bash
conda create -n datasets python=3.10
conda activate datasets
pip install -r requirements.txt
pre-commit install
```

## Datasets
| Dataset       | Use case                |
| ------------- |:-----------------------:|
| CMAPSS        | Time-to-event           |
| CMAPSS        | Discrete time-to-event  |
