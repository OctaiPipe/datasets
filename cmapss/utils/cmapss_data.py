import pandas as pd

def load_data():
    x_train = pd.read_parquet('./data/x_train.parquet.gzip')
    x_test = pd.read_parquet('./data/x_test.parquet.gzip')
    y_train = pd.read_csv('./data/y_train.csv')
    y_test = pd.read_csv('./data/y_test.csv')

    return (x_train, y_train), (x_test, y_test)