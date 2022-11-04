import pandas as pd
import numpy as np

def clean_train_dataf(df, rejected_features):
    def add_train_RUL(grp):
        # Apply clipping to RUL based on uptime
        grp['RUL'] = (grp['uptime'].max() - grp['uptime'] + 1)
        # Apply clipping to RUL
        grp['RUL'].clip(upper=125, inplace=True)
        return grp

    return (df
             .drop(columns=rejected_features)
             .sort_values(by=['machine_number', 'uptime'], axis=0)
             .groupby('machine_number', group_keys=True)
             .apply(add_train_RUL)
            )

def encode_rul(df,
               bins=[1, 40, 80, 125],
               labels=['0', '1', '2']):
    return (df
             .assign(RUL=lambda d: pd.cut(x=d['RUL'],
                                          bins=bins,
                                          labels=labels,
                                          include_lowest=True))
            )


def clean_test_dataf(df, rejected_features):
    def add_test_RUL(grp):
        grp['RUL'] += grp['uptime'].max() - grp['uptime']
        # Apply clipping to RUL
        grp['RUL'].clip(upper=125, inplace=True)
        return grp

    return (df
             .drop(columns=rejected_features)
             .sort_values(['machine_number', 'uptime'], axis=0)
             .groupby('machine_number', group_keys=True)
             .apply(add_test_RUL)
            )

def scale_train_dataf(df):
    from sklearn.preprocessing import MinMaxScaler
    cols = df.columns.difference(['machine_number', 'RUL'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[cols] = scaler.fit_transform(df[cols])

    return df, scaler

def scale_test_dataf(df, scaler):
    cols = df.columns.difference(['machine_number', 'RUL'])
    df[cols] = scaler.transform(df[cols])

    return df

def add_lagged_vars(df, num_lags, columns_not_lagged):
    '''
    Takes a time-series dataframe and appends lagged variables
    using DataFrame.shift().
    '''
    
    # take a copy of the dataframe
    df_copy = df.copy() 
    
    # drop the columns that are not to be lagged
    for column in columns_not_lagged:
        df_copy.drop(column, inplace=True, axis=1)
    
    # get list of shifted dataframes
    shifted_dfs = []
    for i in range(1, num_lags+1):
        shifted_df = df_copy.shift(i)
        shifted_df.columns = [f'{name}_tm{i:02}' for name in df_copy.columns]
        shifted_dfs.append(shifted_df)
        
    # concatenate all the shifted dfs horizontally
    new_df = pd.concat(shifted_dfs, axis=1)

    # now join the original df to the new df
    final_df = pd.concat([df, new_df], axis=1)
    
    # drop rows with nans created from shifting and reset the index
    final_df.dropna(inplace=True)
    final_df.reset_index(inplace=True, drop=True)
    
    return final_df

def lag_dataframe(df, num_lags):
    '''
    Goes through data for each engine unit, and adds lagged variables
    with add_lagged_vars function.
    '''
    
    # get the number of engine units
    num_units = df['machine_number'].iloc[-1]
    
    cols_not_lagged = ['machine_number', 'RUL']
    
    lagged_dfs = []
    for unit in range(1, num_units+1):
        # get the data of the current unit
        unit_df = df.loc[df['machine_number'] == unit]
        
        # add on lagged variables
        unit_df_lagged = add_lagged_vars(unit_df, num_lags, cols_not_lagged)
        lagged_dfs.append(unit_df_lagged)

    df_lagged = pd.concat(lagged_dfs, axis=0) 
    
    return df_lagged

def shape_dataframe_to_sequence(df, num_lags):
    '''
    Goes through each engine unit and reshape it such that
    trims total sequence such that
    num_rows = (n x num_lags) or (n x num_lags + 1) and n is an integer.
    This means that there will be at least 'num_lags' sequence for one
    engine unit.
    Returns:
        numpy.ndarray (num_round_sequences, num_lags, num_sensors)
        where num_round_sequences are full sequences in the dataset
    '''
    rul = []
    subcycles = []
    # Loop over each group
    num_units = int(df.machine_number.max())
    for unit in range(1, num_units+1):
        unit_df = df.loc[df['machine_number'] == unit]
        # Get number of cycles per machine_number
        num_rows = len(unit_df)
        # Remove machine number
        unit_df.pop('machine_number')
        # Pop and keep RUL
        unit_rul = unit_df.pop('RUL').to_numpy()
        unit_df = unit_df.to_numpy()
        # Flip arrays to prioritize ending RUL for sampling
        unit_rul = np.flipud(unit_rul)
        unit_df = np.flipud(unit_df)
        # Get number of subcycles permitted with the specified num_lags
        num_subcycles = num_rows - num_lags + 1
        for row in range(num_subcycles):
            subcycle = unit_df[row:row+num_lags, :]
            rul.append(unit_rul[row])
            subcycles.append(subcycle)

    return subcycles, rul
