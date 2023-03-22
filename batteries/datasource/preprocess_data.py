#!/usr/bin/env python3

# Script to perform basic preprocessing of the Batteries dataset. The aim
# is to generate *.tar.gz files that is compatible for the `datasets` package.
#
# This script will:
#   - Read *.mat and write to *.tar.gz
#
# This script does not:
#   - Add RUL to the dataset
#
# Usage:
#   $ ./preprocess_data.py
#
# Created by: CSN based on implementation by CY
# Last modified: 22/3/2023

import typer
from utils import convert_data_csv

app = typer.Typer()

@app.command()
def main(
    source_dir: str = typer.Option(
        "./raw",
        help="The path to the raw *.mat battery dataset."),
):
    target_dir = "../"
    convert_data_csv(source_dir, target_dir)

    return

if __name__ == "__main__":
    app()
