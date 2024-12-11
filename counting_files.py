import json
from collections import Counter
import os

# Load the JSON data
with open('/home/arsen.abzhanov/Thesis/REPA/ISIC_2018_REPA/train/vae-sd/dataset.json', 'r') as file:
    data = json.load(file)

# Extract the list of lists
labels = data['labels']

# Extract the numbers from the lists
numbers = [label[1] for label in labels]

# Count the occurrences of each number
count = Counter(numbers)

# Print the counts
for number in range(7):
    print(f"Number {number}: {count[number]}")


def count_files_in_subfolders(directory):
    for subdir, dirs, files in os.walk(directory):
        print(f'{subdir}: {len(files)} files')

        

# Example usage
count_files_in_subfolders('/home/arsen.abzhanov/Thesis/ISIC_2018/ISIC_2018_Train_Folder_format')


import numpy as np

# Load the NumPy file (replace 'your_file.npy' with the actual file path)
array = np.load('/home/arsen.abzhanov/Thesis/REPA/ISIC_2018_REPA/train/vae-sd/00008/img-mean-std-00008001.npy')

# Check the shape of the loaded array
print("Shape of the loaded array:", array.shape)