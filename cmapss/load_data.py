import pandas as pd
import os


def load_data():

    # Set the column names for the raw CMAPSS data
    columns = ['machine_number', 'uptime', 'setting_1', 'setting_2', 'setting_3']
    sensor_measurements = [f'sensor_{i:02d}' for i in range(1, 25)]
    columns += sensor_measurements

    # Load train set
    filename = './cmapss/data/train_FD001.txt'

    # Read plain text file
    df_train = pd.read_csv(filename, sep=" ", names=columns, index_col=False)

    # Drop dummy columns
    df_train.drop(columns=['sensor_22', 'sensor_23', 'sensor_24'], inplace=True)

    # Load rejected features for FD001
    rejected_feature_fname = './data/rejected_features.txt'
    if os.path.exists(rejected_feature_fname):
        # Check if has feature selection has been done and load the file
        with open(rejected_feature_fname, 'r') as f:
            rejected_features = f.read().splitlines()

    # Load test set
    df_test = pd.read_csv('./cmapss/data/test_FD001.txt', sep=" ", names=columns, index_col=False)

    # Drop dummy columns
    df_test.drop(columns=['sensor_22', 'sensor_23', 'sensor_24'], inplace=True)

    # Load the y_test
    y_test = pd.read_csv('./data/RUL_FD001.tar.gz', names=['RUL'])
    y_test['machine_number'] = y_test.index + 1

    # Join the test X and label dataframes
    df_test = df_test.join(y_test.set_index('machine_number'), on='machine_number')

    return df_train, df_test
