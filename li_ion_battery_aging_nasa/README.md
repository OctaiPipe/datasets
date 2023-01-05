# Li-Ion Battery Aging Dataset

The Li-Ion Battery Aging Dataset was downloaded from [`NASA Prognostics Center of Excellence data-set repository`](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). For our purposes, it is now hosted in this repository.

Reference: B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

The raw data files are stored in `./data/raw`. This contains the `.mat` files and the Readme files that contain data descriptions and experiment specifications. The raw data files are processed using utility functions `/notebooks/li_io_battery_aging_nasa/utils/utils.py`; see notebooks in /home/`/notebooks/li_io_battery_aging_nasa`. The resulting processed csv files are stored in `./data/processed`. Note that, each battery in each experiment, we have saved the data from the charge, discharge, and impedance measurement cycles into separate csv files.

The sub-folders contains data from batteries in experiment under different sets of conditions (e.g. ambient temperature); potentially useful for studying federated learning.