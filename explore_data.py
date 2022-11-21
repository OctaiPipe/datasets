from datasets import load_dataset

# load_dataset("./cmapss/cmapss.py")
cmapss = load_dataset("text", data_files={"train": "./cmapss/data/RUL_FD001.txt"})
print(type(cmapss))

print(cmapss)