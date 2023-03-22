# Li-Ion Battery Aging Dataset

The Li-Ion Battery Aging Dataset was downloaded from [`NASA Prognostics Center of Excellence data-set repository`](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). For our purposes, it is now hosted in this repository.

Reference: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

## Folder structure
* The raw data files are stored in [`batteries/datasource`](./datasource/). This contains the `.mat` files and the Readme files that contain data descriptions and experiment specifications.
* The raw data files are processed using utility functions `/notebooks/li_io_battery_aging_nasa/utils/utils.py`; see notebooks in /home/`/notebooks/li_io_battery_aging_nasa`.
* The resulting processed csv files are stored in `./0?_BatteryAging*/*.tar.gz` files. Note that, each battery in each experiment, we have saved the data from the charge, discharge, and impedance measurement cycles into separate `*.tar.gz` files (which are compressed `csv` file format).
* The sub-folders contains data from batteries in experiment under different sets of conditions (e.g. ambient temperature); potentially useful for studying federated learning.

## Usage

### Method 1 (recommended)
Install the `datasets` repository as a submodule in your working repository. An example is provided in the [`README`](https://github.com/The-Data-Analysis-Bureau/ml-experiments#readme) of [`ml-experiments`](https://github.com/The-Data-Analysis-Bureau/ml-experiments) repository.

### Method 2
Start by cloning the datasource that you need. Here is a single-line command that you can copy into your shell and checkout the datasource.
``` bash
git clone --depth=1 https://github.com/The-Data-Analysis-Bureau/datasets.git && mv datasets/batteries . && rm -rf datasets datasets/datasource
```
