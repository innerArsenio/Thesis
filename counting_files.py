import json
from collections import Counter, defaultdict
import os
import numpy as np

# Load the JSON data
with open('./BUSI_REPA/vae-sd/dataset.json', 'r') as file:
    data = json.load(file)
# /home/arsen.abzhanov/Thesis_local/REPA/BUSI_REPA/vae-sd/dataset.json
# /home/arsen.abzhanov/Thesis_local/REPA/ISIC_2018_REPA/train/vae-sd/dataset.json
# Extract the list of lists
labels = data['labels']

# Extract the numbers from the lists
numbers = [label[1] for label in labels]

# Count the occurrences of each number
count = Counter(numbers)

# Print the counts
for number in range(3):
    print(f"Number {number}: {count[number]}")


# def count_files_in_subfolders(directory):
#     for subdir, dirs, files in os.walk(directory):
#         print(f'{subdir}: {len(files)} files')

        

# # Example usage
# count_files_in_subfolders('/home/arsen.abzhanov/Thesis_local/ISIC_2018/ISIC_2018_Validation_Folder_format')


# CONCEPT_LABEL_MAP = [
#     [3, 0, 0, 3, 3, 0, 2], # AKIEC
#     [2, 0, 2, 2, 2, 0, 1], # BCC
#     [4, 2, 1, 4, 4, 1, 3], # BKL
#     [5, 1, 1, 5, 5, 1, 0], # DF
#     [0, 0, 0, 0, 0, 0, 0], # MEL
#     [1, 1, 1, 1, 1, 1, 0], # NV
#     [6, 3, 1, 6, 1, 2, 0], # VASC
# ]

# weights = [327, 514, 1099, 115, 1113, 6705, 142]

# # Transpose the matrix to get columns as rows
# transposed_matrix = list(zip(*CONCEPT_LABEL_MAP))

# # Calculate weighted sum of occurrences for each number in each column
# column_weighted_counts = []
# for column in transposed_matrix:
#     weighted_counts = defaultdict(float)
#     for value, weight in zip(column, weights):
#         weighted_counts[value] += weight
#     column_weighted_counts.append(dict(weighted_counts))

# for i, counts in enumerate(column_weighted_counts):
#     print(f"Column {i}: {counts}")


# # Load the NumPy file (replace 'your_file.npy' with the actual file path)
# array = np.load('/home/arsen.abzhanov/Thesis/REPA/ISIC_2018_REPA/train/vae-sd/00008/img-mean-std-00008001.npy')

# # Check the shape of the loaded array
# print("Shape of the loaded array:", array.shape)