# Data loading functions for the HUST bearing dataset
# https://github.com/CHAOZHAO-1/HUSTbearing-dataset

import argparse
import gc
import json
import os
from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd
import yaml

SAMPLING_FREQ_KHZ = 25.641
SKIP_ROWS = 21
DATA_COLUMNS = ['timestamp', 'speed', 'x', 'y', 'z']
TS_COL = 'timestamp'

RAW_DATA_DIR = './data'
OUTPUT_DIR = './data_parsed'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_arguments():

    parser = argparse.ArgumentParser(
        description='data loading and parsing config path')

    parser.add_argument(
        '--config-path',
        '-c',
        action='store',
        dest='config_path',
        help='config path',
        default="./data_parse_config.yml",
        required=False)

    args = parser.parse_args()

    return args


def get_bearing_metadatas(raw_data_dir: str = RAW_DATA_DIR,
                          output_dir: str = OUTPUT_DIR):
    '''Extracts HUST bearing metadata from raw dataset folder structure.

    Args:
        raw_data_dir (str, optional):. Defaults to RAW_DATA_DIR.
        output_dir (str, optional): the metadata is saved to
            <output_dir>/metadata.json.
            Defaults to OUTPUT_DIR.

    Returns:
        metadatas: list of dict, each being the metadata of a bearing
    '''

    metadatas = []
    filenames = os.listdir(raw_data_dir)
    filenames.sort()
    for filename_ in filenames:
        metadata_ = {}
        metadata_['raw_filename'] = filename_
        metadata_['raw_data_path'] = os.path.join(raw_data_dir, filename_)
        fninfo_ = filename_.rstrip('.xls').split('_')
        if fninfo_[0] == '0.5X':
            fninfo_[0] = 'medium'
            fninfo_ = ['M' + '_' + fninfo_[1]] + fninfo_
        elif fninfo_[0] != 'H':
            fninfo_ = ['severe'] + fninfo_
            fninfo_ = ['S' + '_' + fninfo_[1]] + fninfo_
        else:
            fninfo_ = [None] + fninfo_
            fninfo_ = ['H'] + fninfo_
        if fninfo_[3] == 'VS':
            fninfo_ = fninfo_[:3] + ['_'.join(fninfo_[3:])]
        metadata_.update(dict(zip(['fault_label',
                                   'fault_severity', 'fault_location',
                                   'bearing_speed'],
                                  fninfo_)))
        metadata_['sampling_freq_khz'] = SAMPLING_FREQ_KHZ
        metadatas.append(metadata_)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'metadata.json')
            with open(output_path, 'w') as f:
                json.dump(metadatas, f, indent=4)
    return metadatas


def load_parse_xls_to_csv(metadata: dict,
                          output_dir: str = OUTPUT_DIR,):
    '''Loads and parses a bearing MatLab data file (.mat) into pd.DataFrame,
    and saves as csv file.

    Args:
        metadata (dict): metadata of a single bearing,
            from get_bearing_metadatas()
        output_dir (str, optional): if specified, saves the parsed pd.DataFrame
            to csv file.
            Defaults to OUTPUT_DIR.

    Returns:
        bearing_fault_map (dict): same as metadata, with a new field: either
            'parsed_data_path': <output_dir>/<cycle_id_>.csv if output_dir is
            specified, else
            'df': <the parsed dataframe>.
            Also another new field: 'df_length': The CWRU dataset provides one
            ten-second measurement for each sample. So if sampling frequency
            is 12kHz (48kHz), df_length will be ~120000 (480000).
    '''

    bearing_fault_map = deepcopy(metadata)
    # load matlab file with scipy
    filepath = bearing_fault_map['raw_data_path']
    with open(filepath, 'r') as f:
        raw_data = f.read()
        assert raw_data.split('\n')[SKIP_ROWS] == 'Data', 'wrong file format'
        df_ = pd.read_csv(
            StringIO('\n'.join(raw_data.split('\n')[SKIP_ROWS + 1:])),
            delimiter='\t', header=None)
        df_.columns = DATA_COLUMNS
        df_ = df_.sort_values(TS_COL)

    bearing_fault_map['df_length'] = len(df_)
    bearing_fault_map['df_columns'] = DATA_COLUMNS

    bearing_fault_map['sampling_freq_khz'] = ((1 / df_['timestamp'].diff())
                                              .round().value_counts().index[0]
                                              / 1000)

    bearing_fault_map['df'] = df_
    output_path = None
    # if output_dir, save df_ to output_dir/<cycle_id_>.csv,
    # add 'parsed_data_path' to bearing_fault_map and
    # pop out 'df' from bearing_fault_map
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f'{os.path.splitext(bearing_fault_map["raw_filename"])[0]}.csv')
        df_.to_csv(output_path, index=False)
        bearing_fault_map.pop('df')
    bearing_fault_map['parsed_data_path'] = output_path
    gc.collect()

    return bearing_fault_map


def parse_dataset(raw_data_dir: str = RAW_DATA_DIR,
                  output_dir: str = OUTPUT_DIR,
                  **kwargs):
    '''Extracts HUST bearing metadata from raw dataset folder structure,
    then loads and parses a bearing MatLab data file (.mat) into pd.DataFrame,
    and saves as csv file

    Args:
        raw_data_dir (str, optional):. Defaults to RAW_DATA_DIR.
        output_dir (str, optional): if specified, saves the parsed pd.DataFrame
            to csv file.
            Defaults to os.path.join(OUTPUT_DIR, 'csv').

    Returns:
        parsed_metadatas (dict): list of dict, each being the metadata of
            a bearing with a data field: either
            'parsed_data_path': <output_dir>/<cycle_id_>.csv if output_dir is
            specified, else
            'df': <the parsed dataframe>.
            Also another new field: 'df_length': The CWRU dataset provides one
            ten-second measurement for each sample.
    '''
    metadatas = get_bearing_metadatas(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir)

    parsed_metadatas = []
    for metadata in metadatas:
        parsed_metadatas.append(load_parse_xls_to_csv(
            metadata, output_dir=output_dir))
        gc.collect()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'parsed_metadata.json')
        with open(output_path, 'w') as f:
            json.dump(parsed_metadatas, f, indent=4)

    return parsed_metadatas


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # save a copy of the config
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.yml'), 'w') as f:
        yaml.dump(config, f, yaml.SafeDumper)
    _ = config.pop('name')
    parsed_metadatas = parse_dataset(**config)
