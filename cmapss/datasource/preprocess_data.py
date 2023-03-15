#!/usr/bin/env python3

# Script to perform basic preprocessing of the C-MAPSS raw datasets. The aim
# is to generate *.csv files that is compatible for the `datasets` package.
#
# This script will:
#   - Read *.txt and write to *.tar.gz
#   - Join test target labels to test set.
#   - Clip maximum RUL values to 125, cf. Li et al.
#     (https://doi.org/10.1016/j.ress.2017.11.021)
#
# This script does not:
#   - Drop columns, rescale/manipulate features, perform feature selection
#   - Quantise target labels (for classification problem)
#
# Usage:
#   $ ./preprocess_data.py --dataset-id FD001
#
# Created by: CSN
# Last modified: 15/3/2023

import typer
import pandas as pd
from ydata_profiling import ProfileReport

app = typer.Typer()

def complete_ids():
    return ["FD001", "FD002", "FD003", "FD004"]

def add_train_RUL(grp):
    # Apply clipping to RUL based on uptime
    grp["RUL"] = (grp["uptime"].max() - grp["uptime"] + 1)
    # Apply clipping to RUL
    grp["RUL"].clip(upper=125, inplace=True)
    return grp

def add_test_RUL(grp):
    grp["RUL"] += grp["uptime"].max() - grp["uptime"]
    # Apply clipping to RUL
    grp["RUL"].clip(upper=125, inplace=True)
    return grp

@app.command()
def main(
    dataset_id: str = typer.Option(
        "FD001",
        help="The dataset ID to preprocess",
        autocompletion=complete_ids
    )
):

    # Set the column names for the raw CMAPSS data
    columns = ["machine_number", "uptime"]
    settings = [f"setting_{i}" for i in range (1, 4)]
    sensors = [f"sensor_{i:02d}" for i in range(1, 25)]
    columns += settings + sensors

    # Read plain text files
    df_train = pd.read_csv(f"./raw/train_{dataset_id}.txt",
                           sep=" ",
                           names=columns,
                           index_col=False)
    df_test = pd.read_csv(f"./raw/test_{dataset_id}.txt",
                          sep=" ",
                          names=columns,
                          index_col=False)
    y_test = pd.read_csv(f"./raw/RUL_{dataset_id}.txt", names=["RUL"])

    # Set machine number for y_test
    y_test["machine_number"] = y_test.index + 1

    # Join the test X and label dataframes
    df_test = df_test.join(y_test.set_index("machine_number"),
                           on="machine_number")

    # User profiler to detect poor features
    profile = ProfileReport(df_train, minimal=True)
    rejected_features = list(profile.get_rejected_variables())
    rejected_feature_fname = f"../rejected_features_{dataset_id}.txt"
    with open(rejected_feature_fname, 'w') as f:
        for feature in rejected_features:
            f.write(f"{feature}\n")

    # Save files
    print(f"Saving files for {dataset_id}")

    (df_train
      .sort_values(by=["machine_number", "uptime"], axis=0)
      .groupby("machine_number", group_keys=True)
      .apply(add_train_RUL)
    ).to_csv(f"../train_{dataset_id}.tar.gz", index=False)

    (df_test
      .sort_values(["machine_number", "uptime"], axis=0)
      .groupby("machine_number", group_keys=True)
      .apply(add_test_RUL)
    ).to_csv(f"../test_{dataset_id}.tar.gz", index=False)

    print(f"Completed saving files for {dataset_id}")

    # os.link(f"./interim/train_{dataset_id}.tar.gz",
    #         f"../train_{dataset_id}.tar.gz")
    # os.link(f"./interim/test_{dataset_id}.tar.gz",
    #         f"../test_{dataset_id}.tar.gz")

    # Estimate the rejected features from ydata-profiler.
    # The features are not removed from the dataframe, only saved to file.
    # os.link(f"./interim/rejected_features_{dataset_id}.txt",
    #         f"../rejected_features_{dataset_id}.txt")

    return

if __name__ == "__main__":
    app()
