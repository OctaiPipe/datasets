import argparse
import gc
import json
import math
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy
import yaml
from tqdm import tqdm

CYCLE_ID = 'cycle_id'
TS_COL = 'timestamp'
METADATA_FILENAME = 'metadata.json'
SOURCE_DATA_DIR = './data_parsed'
RESAMPLED_OUTPUT_DIR = './data_resampled'
os.makedirs(RESAMPLED_OUTPUT_DIR, exist_ok=True)


def get_arguments():

    parser = argparse.ArgumentParser(
        description='data resampling config path')

    parser.add_argument(
        '--config-path',
        '-c',
        action='store',
        dest='config_path',
        help='config path',
        default="./data_resampling_config.yml",
        required=False)

    args = parser.parse_args()

    return args


def resample_data(source_data_dir: str = SOURCE_DATA_DIR,
                  resampled_output_dir: str = RESAMPLED_OUTPUT_DIR,
                  index_column: str = 'index',
                  data_columns: list = ["DE_time", "FE_time"],
                  signal_duration_sec: float = 10.0,
                  target_fs_khz: float = 12.0,
                  **kwargs):
    '''For CWRU dataset.
    Loads source parsed data. For each bearing cycle, with the
    signal_duration_sec (given in papers), infers the true sampling frequency
    of the cycle. Converts index column into TS_COL='timestamp' column
    accordingly, and then resamples to the specified target sampling frequency.

    Args:
        source_data_dir (str, optional): from the data_parse step.
            Defaults to SOURCE_DATA_DIR.
        resampled_output_dir (str, optional): Defaults to RESAMPLED_OUTPUT_DIR.
        index_column (str, optional): timestep column in the parsed dataframe.
            Defaults to 'index' (for CWRU dataset;
            for other datasets it can be 'timestamp').
        data_columns (list, optional): time series columns for which to compute
            spectrogram/spectrum in parsed data.
            Defaults to ["DE_time", "FE_time"].
        signal_duration_sec (float, optional): signal duration as given by
            dataset decription.
            Defaults to 10.0.
        Target signal sampling rate in kHz (float, optional):
            Defaults to 12.0

    Returns:
        resampled_metadata (dict): same as resampled_metadata
            with new field
            'resampled_data_path': os.path.join(resampled_output_dir,
                                                   f'{cycle_id}.npz')
    '''
    os.makedirs(resampled_output_dir, exist_ok=True)

    # load source metadata
    with open(os.path.join(source_data_dir,
                           METADATA_FILENAME), 'r') as f:
        metadata = json.load(f)
    df_metadata = pd.DataFrame(metadata)

    # infer true sampling frequency from signal duration (df_length) and
    # given fixed signal duration for CWRU dataset
    signal_duration_sec = float(signal_duration_sec)
    target_fs_hz = float(target_fs_khz * 1000)
    df_metadata['inferred_sampling_freq_hz'] = (
        df_metadata['df_length'].astype(int)
        .apply(lambda x: math.floor(x / (signal_duration_sec * 1000)) * 1000))
    metadata = df_metadata.to_dict(orient='records')

    # resample data for each cycle
    resampled_metadata = []
    for cycle_ in tqdm(metadata):
        gc.collect()
        metadata_resampled_ = deepcopy(cycle_)
        cycle_id = cycle_['cycle_id']
        inferred_sampling_freq_hz = \
            metadata_resampled_.pop('inferred_sampling_freq_hz')
        # load cycle's bearing time series from source csv
        df_ = pd.read_csv(cycle_['parsed_data_path'])
        # construct ts_col from inferred sampling frequency and index column
        if index_column in df_ and TS_COL not in df_:
            df_[TS_COL] = df_[index_column].apply(
                lambda x: x / inferred_sampling_freq_hz)
            df_ = df_.drop(index_column, axis=1)
        df_ = df_[[TS_COL] + data_columns]

        # resample dataframe
        x_, t_ = scipy.signal.resample(
            x=df_.drop([TS_COL], axis=1).to_numpy(),
            t=df_[TS_COL].to_numpy(),
            num=int(len(df_) / inferred_sampling_freq_hz * target_fs_hz))
        df_resampled_ = pd.DataFrame(
            np.hstack([np.expand_dims(t_, axis=-1), x_]),
            columns=[TS_COL] + data_columns)
        df_resampled_['cycle_id'] = cycle_id

        # save resampled data into csv
        resampled_data_path_ = os.path.join(resampled_output_dir,
                                            f'{cycle_id}.csv')
        # update metadata
        metadata_resampled_['resampled_data_path'] = \
            resampled_data_path_
        metadata_resampled_['sampling_freq_khz'] = \
            target_fs_khz
        metadata_resampled_['df_length'] = len(df_resampled_)
        metadata_resampled_['df_columns'] = list(df_resampled_.columns)
        df_resampled_.to_csv(resampled_data_path_, index=False)
        resampled_metadata.append(metadata_resampled_)

    # save resampled metadata
    with open(os.path.join(resampled_output_dir, METADATA_FILENAME),
              'w') as f:
        json.dump(resampled_metadata, f, indent=4)
    return resampled_metadata


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # save a copy of the config
    os.makedirs(config['resampled_output_dir'], exist_ok=True)
    with open(os.path.join(config['resampled_output_dir'],
                           'config.yml'), 'w') as f:
        yaml.dump(config, f, yaml.SafeDumper)
    _ = config.pop('name')
    resampling_specs = config.pop('resampling_specs')
    config.update(resampling_specs)
    resampled_metadata = resample_data(**config)
