import datetime
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_mat_to_np(filepath: str):
    '''
    Load battery mat file to structured numpy array containing
    charge, discharge, impedance measurement cycles data
    Args:
        filepath (str)
    Returns:
        cycles (np array): np array storing data from the charge, discharge,
        impedance measurement cycles
    '''
    # get battery number from filename
    battery_num = os.path.basename(filepath).split('.')[0]
    # load mat file into structured numpy array
    mat = loadmat(filepath)
    # get cycles data from battery_num field
    cycles = mat[battery_num]['cycle'][0, 0][0]

    return cycles


def load_charge(cycles):
    cycles_dict = []
    cycle_num = 0
    for n in range(len(cycles)):
        cycle = cycles[n]
        if cycle['type'][0] != 'charge':
            continue
        cycle_num = cycle_num + 1
        ambient_temperature = cycle['ambient_temperature'][0, 0]
        time_start_mat = cycle['time'][0]
        time_start = datetime.datetime(
            int(time_start_mat[0]),
            int(time_start_mat[1]),
            int(time_start_mat[2]),
            int(time_start_mat[3]),
            int(time_start_mat[4])) + \
            datetime.timedelta(seconds=int(time_start_mat[5]))
        data = cycle['data']
        data_len = len(data['Time'][0, 0].T.squeeze())
        for i in range(data_len):
            data_dict = {}
            data_dict['cycle_num'] = cycle_num
            data_dict['ambient_temperature'] = ambient_temperature
            data_dict['cycle_time_start'] = time_start
            for name in data.dtype.names:
                data_dict[name] = data[name][0, 0].T.squeeze()[i]

            cycles_dict.append(data_dict)

    return cycles_dict


def load_discharge(cycles):
    cycles_dict = []
    cycle_num = 0
    for n in range(len(cycles)):
        missing_apacity = False
        cycle = cycles[n]
        if cycle['type'][0] != 'discharge':
            continue
        cycle_num = cycle_num + 1
        ambient_temperature = cycle['ambient_temperature'][0, 0]
        time_start_mat = cycle['time'][0]
        time_start = datetime.datetime(
            int(time_start_mat[0]),
            int(time_start_mat[1]),
            int(time_start_mat[2]),
            int(time_start_mat[3]),
            int(time_start_mat[4])) + \
            datetime.timedelta(seconds=int(time_start_mat[5]))
        data = cycle['data']
        data_len = len(data['Time'][0, 0].T.squeeze())
        for i in range(data_len):
            data_dict = {}
            data_dict['cycle_num'] = cycle_num
            data_dict['ambient_temperature'] = ambient_temperature
            data_dict['cycle_time_start'] = time_start
            # some files have missing Capacity
            try:
                data_dict['Capacity'] = data['Capacity'][0, 0][0, 0]
            except IndexError:
                missing_apacity = True
                data_dict['Capacity'] = np.nan
            for name in data.dtype.names:
                if name != 'Capacity':
                    data_dict[name] = data[name][0, 0].T.squeeze()[i]
                else:
                    pass

            cycles_dict.append(data_dict)
        if missing_apacity:
            print(f"Cycle {n} has missing Capacity")

    return cycles_dict


def load_impedance(cycles):
    cycles_dict = []
    cycle_num = 0
    for n in range(len(cycles)):
        cycle = cycles[n]
        if cycle['type'][0] != 'impedance':
            continue
        cycle_num = cycle_num + 1
        ambient_temperature = cycle['ambient_temperature'][0, 0]
        time_start_mat = cycle['time'][0]
        time_start = datetime.datetime(
            int(time_start_mat[0]),
            int(time_start_mat[1]),
            int(time_start_mat[2]),
            int(time_start_mat[3]),
            int(time_start_mat[4])) + \
            datetime.timedelta(seconds=int(time_start_mat[5]))
        data = cycle['data']
        data_len = len(data['Sense_current'][0, 0].T.squeeze())
        for i in range(data_len):
            data_dict = {}
            data_dict['cycle_num'] = cycle_num
            data_dict['ambient_temperature'] = ambient_temperature
            data_dict['cycle_time_start'] = time_start
            data_dict['Re'] = data['Re'][0, 0][0, 0]
            data_dict['Rct'] = data['Rct'][0, 0][0, 0]
            for name in data.dtype.names:
                if name not in ['Re', 'Rct', 'Rectified_Impedance']:
                    data_dict[name] = data[name][0, 0].T.squeeze()[i]

            cycles_dict.append(data_dict)

    return cycles_dict


def load_cycles_to_df(cycles,
                      skip_charge_cycles: bool = False):
    '''
    Load charge, discharge, impedance measurement cycles from a given file
    into separate pd DataFrames
    Args:
        cycles (structured np array): np array
        storing data from the charge, discharge,
        impedance measurement cycles
        skip_charge_cycles (bool): whether to skip the charge cycles
        (the resulting csv files are a bit larger).
        Defaults to False
    Returns:
        df_charge_cycles, df_discharge_cycles, df_impedance_cycles
        (pd DataFrames). df_charge_cycles is None if skip_charge_cycles
    '''
    if skip_charge_cycles:
        df_charge_cycles = None
    else:
        df_charge_cycles = pd.DataFrame.from_dict(load_charge(cycles))
        df_charge_cycles = df_charge_cycles.drop_duplicates()

    df_discharge_cycles = pd.DataFrame.from_dict(load_discharge(cycles))
    df_impedance_cycles = pd.DataFrame.from_dict(load_impedance(cycles))
    df_discharge_cycles = df_discharge_cycles.drop_duplicates()
    df_impedance_cycles = df_impedance_cycles.drop_duplicates()

    return df_charge_cycles, df_discharge_cycles, df_impedance_cycles


def convert_data_csv(
        raw_dataset_folder='../../li_ion_battery_aging_nasa/data/raw/',
        processed_folder='../../li_ion_battery_aging_nasa/data/processed',
        skip_charge_cycles: bool = False
):
    '''
    Load all raw mat data files from folder and convert them to csv files,
    one for each type of cycle. The output csv files have names of form
    <battery_number>_<cycle>.csv

    Args:
        raw_dataset_folder (str)
        processed_folder (str)
        skip_charge_cycles (bool): whether to skip the charge cycles
        (the resulting csv files are a bit larger)
        Defaults to False
    Returns:

    '''

    error_files = []
    for sub_folder in os.listdir(raw_dataset_folder):
        sub_folder_path = os.path.join(raw_dataset_folder, sub_folder)
        processed_dataset_sub_folder = os.path.join(
            processed_folder, sub_folder)
        os.makedirs(processed_dataset_sub_folder, exist_ok=True)
        mat_files = os.listdir(sub_folder_path)
        mat_files = [
            file for file in mat_files if file.split('.')[-1] == 'mat']

        for mat_file in mat_files:
            filename = mat_file.split('.')[0]
            filepath = os.path.join(sub_folder_path, mat_file)
            print(f'Processing {filepath}')
            try:
                df_charge_cycles, df_discharge_cycles, df_impedance_cycles = \
                    load_cycles_to_df(
                        load_mat_to_np(filepath),
                        skip_charge_cycles)

                if not skip_charge_cycles:
                    df_charge_cycles.to_csv(
                        os.path.join(processed_dataset_sub_folder,
                                     filename + '_charge.csv'),
                        index=False)

                df_discharge_cycles.to_csv(
                    os.path.join(processed_dataset_sub_folder,
                                 filename + '_discharge.csv'),
                    index=False)

                df_impedance_cycles.to_csv(
                    os.path.join(processed_dataset_sub_folder,
                                 filename + '_impedance.csv'),
                    index=False)

            except Exception as e:
                print(e)
                error_files.append(filepath)

    print(error_files)


def add_RUL(df,
            time_column='time',
            piecewise=True,
            initial_RUL=1000000.):
    '''
    Add Remaining-Useful-Life (RUL) target label using the time_column of df

    Args:
        df (pd DataFrame): must contain time_column
        time_column (str): time colume in df with which to construct RUL label
        piecewise (bool): if True, construct piecewise-linear RUL, otherwise
        construct linear RUL.
        Defaults to True
        initial_RUL (float): initial RUL if piecewise.
        Defaults to 1000000.
    Returns:

    '''
    if time_column not in df.columns:
        raise KeyError(f"df does not contain '{time_column}' column")

    df_ = df.copy()
    rul = (df_[time_column].max() - df_[time_column])

    if piecewise:
        rul = np.minimum(rul, initial_RUL)

    df_['RUL'] = rul
    return df_
