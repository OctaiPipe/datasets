# The Case Western Reserve University (CWRU) bearing fault dataset


The [Case Western Reserve University (CWRU) bearing fault dataset](https://engineering.case.edu/bearingdatacenter/) is an academic dataset that contains accelerometer readings from bearings that are normal (healthy) and have faults introduced at various parts in advance (VS run to degradation). Data is collected when the bearings are running at different motor loads. This dataset is a standard benchmark dataset for much of the literature on fault diagnosis.

A detailed discussion is given on [Confluence](https://octaipipe.atlassian.net/wiki/spaces/CAAS/pages/1750532098/Case+Western+Reserve+University+CWRU+bearing+dataset).


References:
- https://engineering.case.edu/bearingdatacenter
- https://github.com/s-whynot/CWRU-dataset
- https://www.sciencedirect.com/science/article/pii/S0888327021010499
- https://arxiv.org/abs/2407.14625
- https://www.mdpi.com/2078-2489/15/5/259
- https://www.sciencedirect.com/science/article/pii/S0888327015002034?ref=cra_js_challenge&fr=RR-1

## The `cwru-bearings` container in `octaipipedatasets` blob storage
The raw dataset is processed in various steps as discussed in the next section. All the corresponding datasets and `metadata.json` are stored under the `cwru-bearings` container in `octaipipedatasets` blob storage.



## Data processing steps and datasets

Data processing steps are implemented in the scripts in `./script` with the corresponding config files.

### 0. The raw dataset
We have downloaded the raw dataset from https://github.com/s-whynot/CWRU-dataset. `metadata.json` is generated using `data_parse.get_cwru_metadatas()`. The raw dataset (in `tar.gz`) and the metadata are stored in the `raw` directory in the blob container.

### 1. Data parsing

`data_parse.py` is run to produce the parsed dataset, which extracts CWRU bearing metadata from raw dataset folder structure, then loads and parses a bearing MatLab data file (.mat) into pd.DataFrame, and saves as csv file. The parsed dataset (in `tar.gz`) and the metadata are stored in the `parsed` directory in the blob container.

### 2. Data resampling

`data_resampling.py` is run, which for each bearing cycle, with the `signal_duration_sec` (given in papers), infers the true sampling frequency of the cycle. It then converts the index column into a timestamp column accordingly, and then resamples to the specified target sampling frequency. The resampled dataset (in `tar.gz`) and the metadata are stored in the `resampled` directory in the blob container.


## 3. Data selection and splitting

Following the above references, `data_selection_split.py`, which selects a subset of the data from the resampled CWRU dataset for model training:

    1. Only use the 12kHz data (before resampling),
        which includes the normal bearing data
    2. Only keep the fault diameters 0.007”, 0.014” and 0.021”
    3. Use all of motor load = 0, 1, 2, 3HP;
        in those papers they dropped the 0HP data.
        We include it to increase data volume
    4. For outer ring faults, the 'OR_position' of '@6' is used
        whenever possible,
        or else use '@3', for each (fault diameters, motor load)

The selected dataset (in `tar.gz`) and the metadata are stored in the `selected` directory in the blob container.

In the same script, the selected then ungoes an FL client split and train-validation-test split:
    1. Find unique values of
        'fl_split_group'=["fault_diameter", "motor_load_hp"].
        Each FL client has all the data of a given value of fl_split_group
    2. Split metadata by fl_split_group
    3. Randomly put normal cycles into the fl clients, so a fl client has
        at most one normal cycle
    4. Train-val-test split on central dataset
    5. Train-val-test split on FL client

The train validation test split cycles in a list of metadata, once cycle for each item in metadata. The split is a time-series split with a given gap, with test set being the most recent, followed by validation set.

The split dataset (in `tar.gz`) and the metadata of each FL client (and central) are stored in the `split` directory in the blob container. In each of these `tar.gz` files, there are the `train`, `val` and `test` folders each containing its csv files (one for each cycle), and a `metadata.json`.


## 3. Feature engineering

spectrogram
TODO
