#!/usr/bin/env python3

from datasets import load_dataset

data_files = {"train": "train_FD001.csv", "test": "test_FD001.csv"}
dataset = load_dataset("./cmapss", data_files=data_files)
print(dataset)