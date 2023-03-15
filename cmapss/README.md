# CMAPSS dataset

* CMAPSS datasets was originally downloaded from [`data.nasa.gov`](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) but has since been redacted. For our purposes, it is now hosted in this repository.

## Folder structure
* The original datasets are found in [`cmapss/datasource`](./datasource/).
* The processed data are the `*.tar.gz` files.
* The rejected columns in `rejected_features_FD*.txt` are suggestions from `ydata-profiler`.
> **Warning**
> The rejected columns need to be reimplemented in your code as they are incompletely detected by `ydata-profiler`. If you're not sure what this means, get in touch with the maintainer of this repo.

## Usage

WIP
