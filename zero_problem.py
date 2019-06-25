import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time


print('------ Data loading... ------')

dataset = pd.read_csv('input.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
dataset_labels = pd.read_csv('labels.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.

total_count = dataset_labels.shape[0]

unique, counts = np.unique(dataset_labels, return_counts=True)
print(dict(zip(unique, counts)))


zero_indieces = []

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


dataset = np.delete(dataset, [3, 4, 26, 27, 49, 50, 72, 73, 95, 96, 118, 119, 141, 142, 164, 165, 187, 188, 210, 211], 1)
np.savetxt("zero_input.csv", dataset, delimiter=",")
np.savetxt("zero_labels.csv", dataset_labels, delimiter=",")
