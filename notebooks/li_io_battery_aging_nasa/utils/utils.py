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


def load_cycles_to_df(cycles):
    '''
    Load charge, discharge, impedance measurement cycles from a given file
    into separate pd DataFrames
    Args:
        cycles (structured np array): np array
        storing data from the charge, discharge,
        impedance measurement cycles
    Returns:
        df_charge_cycles, df_discharge_cycles, df_impedance_cycles
        (pd DataFrames)
    '''

    df_charge_cycles = pd.DataFrame.from_dict(load_charge(cycles))
    df_discharge_cycles = pd.DataFrame.from_dict(load_discharge(cycles))
    df_impedance_cycles = pd.DataFrame.from_dict(load_impedance(cycles))

    return df_charge_cycles, df_discharge_cycles, df_impedance_cycles
