import numpy as np
import pandas as pd

WINDOW_SIZE = 10
INPUT_FEATURES = 23

print('------ Data loading... ------')

dataset = pd.read_csv('step_1C.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
dataset_labels = pd.read_csv('step_2C.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.

total_count = dataset_labels.shape[0]

unique, counts = np.unique(dataset_labels, return_counts=True)
print(dict(zip(unique, counts)))

features_for_delete = []

for counter in range(3, INPUT_FEATURES * WINDOW_SIZE, INPUT_FEATURES):
    features_for_delete.append(counter)
    features_for_delete.append(counter + 1)

dataset = np.delete(dataset, features_for_delete, 1)
np.savetxt("step_1C.csv", dataset, delimiter=",")
np.savetxt("step_2C.csv", dataset_labels, delimiter=",")
