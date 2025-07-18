import argparse
import gc
import json
import os
import shutil
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

CYCLE_ID = 'cycle_id'
TS_COL = 'timestamp'
METADATA_FILENAME = 'metadata.json'
SOURCE_DATA_DIR = './data_resampled'
SOURCE_FILEPATH_FIELD = 'resampled_data_path'
SELECTED_OUTPUT_DIR = './data_selected'
SELECTED_FILEPATH_FIELD = 'selected_data_path'
os.makedirs(SELECTED_OUTPUT_DIR, exist_ok=True)
SPLIT_OUTPUT_DIR = './data_split'
SPLIT_METADATA_FILENAME = 'split_metadata.json'
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)


def get_arguments():

    parser = argparse.ArgumentParser(
        description='data selection and split config path')

    parser.add_argument(
        '--config-path',
        '-c',
        action='store',
        dest='config_path',
        help='config path',
        default="./data_selection_split_config.yml",
        required=False)

    args = parser.parse_args()

    return args


def select_cwru_data(
        source_data_dir: str = SOURCE_DATA_DIR,
        source_filepath_field: str = SOURCE_FILEPATH_FIELD,
        selected_output_dir: str = SELECTED_OUTPUT_DIR,
        drop_48k: bool = True,
        fault_diameters_to_keep: list = [None, "007", "014", "021"],
        motor_loads_to_keep: list = ["0", "1", "2", "3"],
        OR_use_6_alt_3: bool = True,
        **kwargs):
    '''For CWRU dataset.
    Select data from parsed, resampled CWRU dataset for model training.
    Follows:
    https://www.sciencedirect.com/science/article/pii/S0888327021010499
    https://arxiv.org/abs/2407.14625

    1. Only use the 12kHz data (before resampling),
        which includes the normal bearing data
    2. Only keep the fault diameters 0.007”, 0.014” and 0.021”
    3. Use all of motor load = 0, 1, 2, 3HP;
        in those papers they dropped the 0HP data.
        We include it to increase data volume
    4. For outer ring faults, the 'OR_position' of '@6' is used
        whenever possible,
        or else use '@3', for each (fault diameters, motor load)

    To the output directory we copy the selected cycles' csv
    and only keep the corresponding records in the metadata json.

    Args:
        source_data_dir (str, optional): from the data_parse step.
            Defaults to SOURCE_DATA_DIR.
        source_filepath_field (str, optional): field in the metadata
            pointing to the source filepath
            Defaults to SOURCE_FILEPATH_FIELD.
        selected_output_dir (str, optional): Defaults to SELECTED_OUTPUT_DIR.
        drop_48k (bool): drop 48kHz (before resampling) cycles.
            Defaults to True
        fault_diameters_to_keep: defaults to [None, "007", "014", "021"],
        motor_loads_to_keep: defaults to ["0", "1", "2", "3"]
        OR_use_6_alt_3: for outer ring (OR) fault, only use @6 o'clock
            cycles, or use @3 o'clock if the former is absent.
            Defaults to True.

    Returns:
        selected_metadata (dict):
    '''
    os.makedirs(selected_output_dir, exist_ok=True)

    # load source metadata
    with open(os.path.join(source_data_dir,
                           METADATA_FILENAME), 'r') as f:
        metadata = json.load(f)
    df_metadata = pd.DataFrame(metadata)

    # drop 48kHz (before resampling) data
    if drop_48k:
        ind_48k = [ind for ind, path in enumerate(
            df_metadata['raw_data_path'].to_list())
            if '48k' in path]
        df_metadata_selected = df_metadata.iloc[
            ~df_metadata.index.isin(ind_48k)]

    # select fault diameters and motor loads to keep
    df_metadata_selected = df_metadata_selected[
        (df_metadata_selected['fault_diameter']
         .isin(fault_diameters_to_keep)) &
        (df_metadata_selected['motor_load_hp']
         .isin(motor_loads_to_keep))]

    df_metadata_selected_OR = df_metadata_selected[
        df_metadata_selected['fault_location'] == 'OR']
    df_metadata_selected_not_OR = df_metadata_selected[
        df_metadata_selected['fault_location'] != 'OR']

    # use @6 OR positions, or @3 if not exist
    if OR_use_6_alt_3:
        group_list = [(x, y, z)
                      for x in fault_diameters_to_keep
                      if x is not None
                      for y in motor_loads_to_keep
                      for z in ['DE', 'FE']]
        OR_cycles_to_keep = []
        for (diam, motor_load, ftype) in group_list:
            df_ = \
                df_metadata_selected_OR[
                    (df_metadata_selected_OR['fault_diameter'] == diam) &
                    (df_metadata_selected_OR['fault_type'] == ftype) &
                    (df_metadata_selected_OR['motor_load_hp'] == motor_load)]
            or_positions = df_['OR_position'].unique()
            if '@6' in or_positions:
                OR_cycles_to_keep.append(
                    df_[df_['OR_position'] == '@6']['cycle_id'].iloc[0])
            elif '@3' in or_positions:
                OR_cycles_to_keep.append(
                    df_[df_['OR_position'] == '@3']['cycle_id'].iloc[0])
        df_metadata_selected_OR = df_metadata_selected_OR[
            df_metadata_selected_OR['cycle_id'].isin(OR_cycles_to_keep)]

    df_metadata_selected = pd.concat(
        [df_metadata_selected_not_OR, df_metadata_selected_OR])
    df_metadata_selected['selected_data_path'] = \
        df_metadata_selected['cycle_id'].astype(str) + '.csv'
    df_metadata_selected['selected_data_path'] = \
        df_metadata_selected['selected_data_path'].apply(
            lambda x: os.path.join(selected_output_dir, x))
    selected_metadata = df_metadata_selected.to_dict(orient='records')

    # save selected data metadata
    with open(os.path.join(selected_output_dir, METADATA_FILENAME),
              'w') as f:
        json.dump(selected_metadata, f, indent=4)

    # copy selected datafiles to output dir
    paths_to_copy = df_metadata_selected[source_filepath_field].to_list()
    dest_paths = [os.path.join(selected_output_dir, os.path.split(path)[-1])
                  for path in paths_to_copy]
    for src_path, dest_path in dict(zip(paths_to_copy, dest_paths)).items():
        shutil.copy2(src_path, dest_path)
    return selected_metadata


def train_test_split(metadata: list,
                     output_dir: str,
                     test_size: float = 0.2, gap: int = 10,
                     filepath_field: str = SELECTED_FILEPATH_FIELD,
                     ts_col: str = TS_COL,
                     train_dir: str = 'train',
                     val_dir: str = 'val',
                     test_dir: str = 'test'):
    '''Train validation test split cycles in a list of metadata, once cycle for
    each item in metadata. The split is a time-series split with a given gap,
    with test set being the most recent, followed by validation set.

    Args:
        metadata (list): list of cycle metadata
        output_dir (str):
        test_size (float, optional): test set size in fraction.
            Defaults to 0.2.
        gap (int, optional): gap between train, val and test sets to avoid
            data leakage.
            Defaults to 10.
        filepath_field (str, optional): field in metadata
            pointing to the source csv file.
            Defaults to SELECTED_FILEPATH_FIELD.
        ts_col (str, optional): timestamp column in csv file.
            Defaults to TS_COL.
        train_dir (str, optional): output train sub-dir. Defaults to 'train'.
        val_dir (str, optional): output val sub-dir. Defaults to 'val'.
        test_dir (str, optional): output test sub-dir. Defaults to 'test'.

    Returns:
        metadata_split: metadata after train-val-test split:
            has new fields 'test_data_path', 'val_data_path', 'train_data_path'
    '''
    os.makedirs(output_dir, exist_ok=True)

    test_dir = os.path.join(output_dir, test_dir)
    os.makedirs(test_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, val_dir)
    os.makedirs(val_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, train_dir)
    os.makedirs(train_dir, exist_ok=True)

    metadata_split = deepcopy(metadata)
    for n, met in enumerate(metadata_split):
        cycle_id = met[CYCLE_ID]
        data_path = met[filepath_field]
        df_ = pd.read_csv(data_path).sort_values(ts_col)
        df_test = df_.iloc[- int(test_size * len(df_)):]
        df_val = df_.iloc[int((1 - 2 * test_size) * len(df_)) - gap:
                          int((1 - 2 * test_size) * len(df_)) - gap
                          + int(test_size * len(df_))]
        df_train = df_.iloc[:int((1 - 2 * test_size) * len(df_)) - 2 * gap]

        test_data_path = os.path.join(test_dir, f'{cycle_id}.csv')
        val_data_path = os.path.join(val_dir, f'{cycle_id}.csv')
        train_data_path = os.path.join(train_dir, f'{cycle_id}.csv')
        metadata_split[n]['test_data_path'] = test_data_path
        metadata_split[n]['val_data_path'] = val_data_path
        metadata_split[n]['train_data_path'] = train_data_path

        df_test.to_csv(test_data_path, index=False)
        df_val.to_csv(val_data_path, index=False)
        df_train.to_csv(train_data_path, index=False)

    return metadata_split


def split_cwru_data(selected_output_dir: str = SELECTED_OUTPUT_DIR,
                    split_output_dir: str = SPLIT_OUTPUT_DIR,
                    selected_filepath_field: str = SELECTED_FILEPATH_FIELD,
                    fl_split_group: list = ["fault_diameter", "motor_load_hp"],
                    test_size: float = 0.2,
                    gap: int = 10,
                    random_state: int = 42,
                    **kwargs):
    '''For CWRU dataset.
    Split dataset into FL clients, and do train-val-test split.
    Loosely follows:
    https://www.sciencedirect.com/science/article/pii/S0888327021010499
    https://arxiv.org/abs/2407.14625

    1. Find unique values of
        'fl_split_group'=["fault_diameter", "motor_load_hp"].
        Each FL client has all the data of a given value of fl_split_group
    2. Split metadata by fl_split_group
    3. Randomly put normal cycles into the fl clients, so a fl client has
        at most one normal cycle
    4. Train-val-test split on central dataset
    5. Train-val-test split on FL client

    To the output directory we copy the selected cycles' csv
    and only keep the corresponding records in the metadata json.

    Args:
        selected_output_dir (str, optional): Defaults to SELECTED_OUTPUT_DIR.
        split_output_dir (str, optional): Defaults to SPLIT_OUTPUT_DIR.
        fl_split_group: FL client dataset split is done by this group.
        test_size (float, optional): test set size in fraction.
            Defaults to 0.2.
        gap (int, optional): gap between train, val and test sets to avoid
            data leakage.
            Defaults to 10.
        random_state: used to assign normal bearing cycles to FL clients

    Returns:
        selected_metadata (dict):
    '''
    prng = np.random.default_rng(random_state)

    os.makedirs(selected_output_dir, exist_ok=True)

    # load source metadata
    with open(os.path.join(selected_output_dir,
                           METADATA_FILENAME), 'r') as f:
        metadata = json.load(f)
    df_metadata = pd.DataFrame(metadata)

    # separate out normal cycles
    normal_cycles = df_metadata[df_metadata['fault_type'] == 'N']
    fault_cycles = df_metadata[df_metadata['fault_type'] != 'N']

    # split metadatas by fl_split_group
    fault_cycles = fault_cycles.set_index(fl_split_group).sort_index()
    fl_df_metadatas = []
    for ind in fault_cycles.index.unique():
        df_ = fault_cycles.loc[ind].reset_index()
        df_ = df_[normal_cycles.columns]
        fl_df_metadatas.append(fault_cycles.loc[ind].reset_index())

    # randomly put normal cycles into the fl clients, so a fl client has
    # at most one normal cycle
    clients_normal = prng.choice(len(fl_df_metadatas),
                                 len(normal_cycles), replace=False)

    for i, n in enumerate(clients_normal):
        fl_df_metadatas[n] = pd.concat([fl_df_metadatas[n],
                                        normal_cycles.iloc[[i]]])

    # train-test split on central dataset
    central_metadatas = [df_.to_dict(orient='records')
                         for df_ in fl_df_metadatas]
    central_metadatas = [item for sublist in central_metadatas
                         for item in sublist]
    central_dir = os.path.join(split_output_dir, 'central')
    os.makedirs(central_dir, exist_ok=True)
    central_metadatas_split = train_test_split(
        metadata=central_metadatas,
        output_dir=central_dir,
        test_size=test_size, gap=gap, ts_col=TS_COL,
        filepath_field=selected_filepath_field
    )

    with open(os.path.join(central_dir, METADATA_FILENAME),
              'w') as f:
        json.dump(central_metadatas_split, f, indent=4)

    # train-test split on FL datasets
    fl_metadatas = [df_.to_dict(orient='records')
                    for df_ in fl_df_metadatas]
    with open(os.path.join(split_output_dir, 'fl_metadata.json'),
              'w') as f:
        json.dump(dict(zip([f'client_{n}' for n in range(len(fl_metadatas))],
                           fl_metadatas)), f, indent=4)
    for n, client_metadata in enumerate(fl_metadatas):
        # save selected data metadata
        client_dir = os.path.join(split_output_dir, f'client_{n}')
        os.makedirs(client_dir, exist_ok=True)
        client_metadata_split = train_test_split(
            metadata=client_metadata,
            output_dir=client_dir,
            test_size=test_size, gap=gap, ts_col=TS_COL,
            filepath_field=selected_filepath_field)
        with open(os.path.join(client_dir, METADATA_FILENAME), 'w') as f:
            json.dump(client_metadata_split, f, indent=4)


if __name__ == "__main__":
    # load arguments
    args = get_arguments()
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    # save a copy of the config
    os.makedirs(config['selected_output_dir'], exist_ok=True)
    with open(os.path.join(config['selected_output_dir'], 'config.yml'),
              'w') as f:
        yaml.dump(config, f, yaml.SafeDumper)
    _ = config.pop('name')
    selection_config = deepcopy(config)
    selection_specs = selection_config.pop('selection_specs')
    selection_config.update(selection_specs)
    selected_metadata = select_cwru_data(**selection_config)
    split_config = deepcopy(config)
    split_specs = split_config.pop('split_specs')
    split_config.update(split_specs)
    split_metadata = split_cwru_data(**split_config)
