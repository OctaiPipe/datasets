#!/usr/bin/env python3

from datasets import load_dataset

# load_dataset("./cmapss/cmapss.py")
# cmapss = load_dataset("text", data_files={"train": "./cmapss/data/RUL_FD001.txt"})
# print(type(cmapss))
# 
# print(cmapss)
# dataset = load_dataset("./cmapss")

data_files = {"train": "train_FD001.csv", "test": "test_FD001.csv"}
dataset = load_dataset("./cmapss", data_files=data_files)
print(dataset)