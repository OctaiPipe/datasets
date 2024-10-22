# Data loading functions for the Case Wester Reserve University (CWRU)
# bearing dataset
# https://engineering.case.edu/bearingdatacenter/welcome

import argparse
import gc
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat

CYCLE_ID = 'cycle_id'
RAW_DATA_DIR = './data'
PARSED_OUTPUT_DIR = './data_parsed'
METADATA_FILENAME = 'metadata.json'
os.makedirs(PARSED_OUTPUT_DIR, exist_ok=True)


NORMAL_SAMPLING_FREQ_KHZ = 12.0

FAULT_TYPE_MAP = {
    'Normal': 'N',
    '12k_Drive_End_Bearing_Fault_Data': 'DE',
    '12k_Fan_End_Bearing_Fault_Data': 'FE',
    '48k_Drive_End_Bearing_Fault_Data': 'DE'
}
FAULT_TYPE_FOLDERS = list(FAULT_TYPE_MAP.keys())
FAULT_FREQ_MAP = {
    'Normal': 12.0,
    '12k_Drive_End_Bearing_Fault_Data': 12.0,
    '12k_Fan_End_Bearing_Fault_Data': 12.0,
    '48k_Drive_End_Bearing_Fault_Data': 48.0
}
FAULT_LOCATIONS = ['B', 'IR', 'OR']
FAULT_DIAMETERS = ['007', '014', '021', '028', '040']
OR_POSITIONS = ['@3', '@6', '@12']


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


def get_cwru_metadatas(raw_data_dir: str = RAW_DATA_DIR,
                       output_dir: str = OUTPUT_DIR):
    '''Extracts CWRU bearing metadata from raw dataset folder structure.

    Args:
        raw_data_dir (str, optional):. Defaults to RAW_DATA_DIR.
        output_dir (str, optional): the metadata is saved to
            <output_dir>/metadata.json.
            Defaults to OUTPUT_DIR.

    Returns:
        metadatas: list of dict, each being the metadata of a bearing
    '''
    metadatas = []
    for fault_type_folder in FAULT_TYPE_FOLDERS:
        fault_label = None
        sampling_freq_khz = FAULT_FREQ_MAP[fault_type_folder]

        if fault_type_folder == 'Normal':
            # get normal bearing metadata from folder structure
            fault_label = 'N'
            filenames = os.listdir(os.path.join(raw_data_dir,
                                                fault_type_folder))
            for filename in filenames:
                metadatas.append({
                    'raw_data_path': os.path.join(
                        raw_data_dir, fault_type_folder, filename),
                    CYCLE_ID: filename.split('_')[0],
                    'fault_label': fault_label,
                    'motor_load_hp': filename.split('_')[-1].rstrip('.mat'),
                    'sampling_freq_khz': sampling_freq_khz,
                    'fault_type': FAULT_TYPE_MAP[fault_type_folder],
                    'fault_location': None,
                    'fault_diameter': None,
                    'OR_position': None})
            continue
        for fault_location in FAULT_LOCATIONS:
            # get faulty bearing metadata from folder structure

            if not os.path.exists(os.path.join(
                    raw_data_dir, fault_type_folder, fault_location)):
                continue
            for fault_diameter in FAULT_DIAMETERS:
                if not os.path.exists(os.path.join(
                        raw_data_dir, fault_type_folder, fault_location,
                        fault_diameter)):
                    continue
                if fault_location != 'OR':
                    filenames = os.listdir(os.path.join(raw_data_dir,
                                                        fault_type_folder,
                                                        fault_location,
                                                        fault_diameter))
                    fault_type = FAULT_TYPE_MAP[fault_type_folder]
                    fault_label = fault_type + '_' + fault_location
                    for filename in filenames:
                        metadatas.append({
                            'raw_data_path': os.path.join(
                                raw_data_dir, fault_type_folder,
                                fault_location,
                                fault_diameter,
                                filename),
                            CYCLE_ID: filename.split('_')[0],
                            'fault_label': fault_label,
                            'motor_load_hp': (filename.split('_')[1]
                                              .rstrip('.mat')),
                            'sampling_freq_khz': sampling_freq_khz,
                            'fault_type': fault_type,
                            'fault_location': fault_location,
                            'fault_diameter': fault_diameter,
                            'OR_position': None})
                else:
                    for OR_position in OR_POSITIONS:
                        # outer ring (OR) fault is further subdivided by
                        # fault position
                        if not os.path.exists(os.path.join(
                                raw_data_dir, fault_type_folder,
                                fault_location,
                                fault_diameter, OR_position)):
                            continue
                        filenames = os.listdir(os.path.join(raw_data_dir,
                                                            fault_type_folder,
                                                            fault_location,
                                                            fault_diameter,
                                                            OR_position))
                        for filename in filenames:
                            fault_type = FAULT_TYPE_MAP[fault_type_folder]
                            fault_label = (fault_type + '_' + fault_location
                                        #    + OR_position
                                           )
                            metadatas.append({
                                'raw_data_path': os.path.join(
                                    raw_data_dir, fault_type_folder,
                                    fault_location,
                                    fault_diameter,
                                    OR_position,
                                    filename),
                                CYCLE_ID: (filename.split('_')[0]
                                             .split('@')[0]),
                                'fault_label': fault_label,
                                'motor_load_hp': (filename.split('_')[1]
                                                  .rstrip('.mat')),
                                'sampling_freq_khz': sampling_freq_khz,
                                'fault_type': fault_type,
                                'fault_location': fault_location,
                                'fault_diameter': fault_diameter,
                                'OR_position': OR_position})
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metadata.json')
        with open(output_path, 'w') as f:
            json.dump(metadatas, f, indent=4)
    return metadatas


def load_parse_cwru_mat_to_csv(metadata: dict,
                               output_dir: str = OUTPUT_DIR):
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
    mat_dict_ = loadmat(bearing_fault_map['raw_data_path'])
    mat_dict_keys_ = list(mat_dict_.keys())
    cycle_id_ = bearing_fault_map[CYCLE_ID]
    # get RPM from matlab file
    bearing_fault_map['rpm'] = None
    match cycle_id_:
        case '97':
            bearing_fault_map['rpm'] = 1797
        case '98':
            bearing_fault_map['rpm'] = 1772
        case '99':
            bearing_fault_map['rpm'] = 1750
        case '100':
            bearing_fault_map['rpm'] = 1730
        case _:
            RPM_key = [key for key in mat_dict_keys_
                       if 'RPM' in key]
            if len(RPM_key) > 1:
                RPM_key = [key for key in RPM_key if cycle_id_ in key]
            if len(RPM_key) > 0:
                rpm = int(mat_dict_[RPM_key[0]].squeeze())
                bearing_fault_map['rpm'] = rpm

    df_ = pd.DataFrame([])
    df_columns = []

    for df_key_ in ['DE_time', 'FE_time', 'BA_time']:
        mat_dict_key = [key for key in mat_dict_keys_
                        if df_key_ in key]

        # some matlab files have data of multiple bearings labelled by key.
        # only load the data corresponding to that bearing indicated by
        # the filename.
        if len(mat_dict_key) > 1:
            mat_dict_key = [key for key in mat_dict_key
                            if cycle_id_ in key]
        if len(mat_dict_key) > 0:
            df_[df_key_] = mat_dict_[mat_dict_key[0]].squeeze()
            df_columns.append(df_key_)
        else:
            df_[df_key_] = None
    df_[CYCLE_ID] = cycle_id_
    df_columns.append(CYCLE_ID)
    bearing_fault_map['df_length'] = len(df_)
    bearing_fault_map['df_columns'] = df_columns

    bearing_fault_map['df'] = df_
    output_path = None
    # if output_dir, save df_ to output_dir/<cycle_id_>.csv,
    # add 'parsed_data_path' to bearing_fault_map and
    # pop out 'df' from bearing_fault_map
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{cycle_id_}.csv')
        df_.to_csv(output_path, index=True, index_label='index')
        bearing_fault_map.pop('df')
    bearing_fault_map['parsed_data_path'] = output_path
    gc.collect()

    return bearing_fault_map


def parse_cwru_dataset(raw_data_dir: str = RAW_DATA_DIR,
                       parse_output_dir: str =
                       os.path.join(PARSED_OUTPUT_DIR, 'csv'),
                       **kwargs):
    '''Extracts CWRU bearing metadata from raw dataset folder structure,
    then loads and parses a bearing MatLab data file (.mat) into pd.DataFrame,
    and saves as csv file

    Args:
        raw_data_dir (str, optional):. Defaults to RAW_DATA_DIR.
        parse_output_dir (str, optional): if specified, saves the
            parsed pd.DataFrame
            to csv file.
            Defaults to os.path.join(PARSED_OUTPUT_DIR, 'csv').

    Returns:
        parsed_metadatas (dict): list of dict, each being the metadata of
            a bearing with a data field: either
            'parsed_data_path': <parse_output_dir>/<cycle_id_>.csv
            if parse_output_dir is
            specified, else
            'df': <the parsed dataframe>.
            Also another new field: 'df_length': The CWRU dataset provides one
            ten-second measurement for each sample. So if sampling frequency
            is 12kHz (48kHz), df_length will be ~120000 (480000).
    '''
    metadatas = get_cwru_metadatas(
        raw_data_dir=raw_data_dir,
        output_dir=parse_output_dir)

    parsed_metadatas = []
    for metadata in metadatas:
        parsed_metadatas.append(load_parse_cwru_mat_to_csv(
            metadata, output_dir=parse_output_dir))
        gc.collect()

    if parse_output_dir:
        os.makedirs(parse_output_dir, exist_ok=True)
        output_path = os.path.join(parse_output_dir, METADATA_FILENAME)
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
    os.makedirs(config['parse_output_dir'], exist_ok=True)
    with open(os.path.join(config['parse_output_dir'],
                           'config.yml'), 'w') as f:
        yaml.dump(config, f, yaml.SafeDumper)
    _ = config.pop('name')
    parsed_metadatas = parse_cwru_dataset(**config)
