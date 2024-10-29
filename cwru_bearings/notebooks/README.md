# The Case Western Reserve University (CWRU) bearing fault dataset


The [Case Western Reserve University (CWRU) bearing fault dataset](https://engineering.case.edu/bearingdatacenter/) is an academic dataset that contains accelerometer readings from bearings that are normal (healthy) and have faults introduced at various parts in advance (VS run to degradation). Data is collected when the bearings are running at different motor loads. This dataset is a standard benchmark dataset for much of the literature on fault diagnosis.

A detailed discussion is given on [Confluence](https://octaipipe.atlassian.net/wiki/spaces/CAAS/pages/1796866054/Data+pipeline+for+Case+Western+Reserve+University+CWRU+bearing+dataset).


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

To run:
1. Fill out `data_parse_config.yml`
2. `python3 data_parse.py`

### 2. Data resampling

`data_resampling.py` is run, which for each bearing cycle, with the `signal_duration_sec` (given in papers), infers the true sampling frequency of the cycle. It then converts the index column into a timestamp column accordingly, and then resamples to the specified target sampling frequency. The resampled dataset (in `tar.gz`) and the metadata are stored in the `resampled` directory in the blob container.

To run:
1. Fill out `data_resampling_config.yml`
2. `python3 data_resampling.py`  (points to default config file `data_resampling_config.yml`)

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

To run:
1. Fill out `data_selection_split_config.yml`
2. `python3 data_selection_split.py`

## 4. Preprocessing (time series sequences, 1d FFT, 2d spectrograms)


Following the above references, `data_preprocessing.py` generates time series sequences, 1d FFT, 2d spectrograms.

Option 1: Run preprocessing on a given FL client folder and a subset (train/val/test)
To run:
1. Fill out `data_preprocessing_config.yml`
    In particular, specify data source, e.g.
```
source_data_dir: ./data_split/client_0
source_filepath_field: "train_data_path"
fftspectrum_output_dir: ./data_fftspectrum/client_0/train
fftspectrum_output_filepath_field: train_fftspectrum_data_path
spectrogram_output_dir: ./data_spectrogram/client_0/train
spectrogram_output_filepath_field: train_spectrogram_data_path
```
2. `python3 data_preprocessing.py` (points to `data_preprocessing_config.yml` by default)


Option 2: Batch run preprocessing on all FL client/central folders and subsets
To run:
1. Fill out `data_preprocessing_config_template.yml`
    In particular, specify data source containing all FL client/central folders and subsets, e.g.
```
source_data_dir: ./data_split
source_filepath_field: data_path
rawts_output_dir: ./data_rawts
rawts_output_filepath_field: rawts_data_path
fftspectrum_output_dir: ./data_fftspectrum
fftspectrum_output_filepath_field: fftspectrum_data_path
spectrogram_output_dir: ./data_spectrogram
spectrogram_output_filepath_field: spectrogram_data_path
```
2. `python3 data_preprocessing.py -b` (`b` stands for batch run. points to `data_preprocessing_config_template.yml` by default)

