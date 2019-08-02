import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time

WINDOW_SIZE = 9
print('------ Data loading... ------')

dataset = pd.read_csv('step_1.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
dataset_labels = pd.read_csv('step_2.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.

total_count = dataset_labels.shape[0]

unique, counts = np.unique(dataset_labels, return_counts=True)
print(dict(zip(unique, counts)))


zero_indieces = []
features_for_delete = []

for index, (label) in enumerate(dataset_labels):
    label = int(label[0])

    if label == 0:
        zero_indieces.append(index)

    if index % 10000 == 0:
        print((index/total_count) * 100)

dataset = np.delete(dataset, zero_indieces, 0)
dataset_labels = np.delete(dataset_labels, zero_indieces, 0)

unique, counts = np.unique(dataset_labels, return_counts=True)
print(dict(zip(unique, counts)))


for counter in range(3, 23 * WINDOW_SIZE, 23):
    features_for_delete.append(counter)
    features_for_delete.append(counter + 1)

dataset = np.delete(dataset, features_for_delete, 1)
np.savetxt("step_1.csv", dataset, delimiter=",")
np.savetxt("step_1_lb.csv", dataset_labels, delimiter=",")
