# Lets start with the imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time

WINDOWS_SIZE = 10
WINDOWS_STEP = 5

print('------ Data loading... ------')
# Create testset andnn.Linear(128, 256), dataset
dataset = pd.read_csv('subject_1.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
li = dataset
dataset = pd.read_csv('subject_2.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_3.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_4.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_5.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_6.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_7.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_8.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_9.txt', delimiter=" ", header=None, dtype=np.float32).values
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_10.txt', delimiter=" ", header=None, dtype=np.float32).values
dataset = np.append(li, dataset, axis=0)

# Inint scaler for Normalization
min_max_scaler = preprocessing.MinMaxScaler()

if __name__ == '__main__':
    labels = dataset[:, 23:]
    inputs = min_max_scaler.fit_transform(dataset[:, :23])  # Normalization
    input_neurons = np.arange(230).reshape(1, 230)
    input_neurons_labels = np.array([])

    for i in range(0, inputs.shape[0], WINDOWS_STEP):

        # Getting min for not to pass boundaries
        max_size = min(i + WINDOWS_SIZE, inputs.shape[0])

        # get one 'Window'
        buffer = inputs[[x for x in range(i, max_size)]]

        # ---------- Working with labels ----------

        # init label for current window
        window_label = 0

        # get window labels
        buffer_labels = labels[[x for x in range(i, max_size)]]

        # Getting MAX and MIN labels
        max_label = max(buffer_labels)
        min_label = min(buffer_labels)

        max_label_count = np.count_nonzero(buffer_labels == max_label)

        if max_label_count >= WINDOWS_SIZE / 2:
            window_label = max_label
        else:
            window_label = min_label

        # ---------- Working with labels -----------

        # Reshape array for input neurons
        buffer = buffer.reshape(1, -1)

        if buffer.shape[1] >= 230:
            try:
                input_neurons = np.vstack((input_neurons, buffer))
                input_neurons_labels = np.concatenate((input_neurons_labels, window_label))
            except:
                print(buffer.shape)

        if i % 100000 == 0:
            print("Processing...")

    print(input_neurons.shape)
    np.savetxt("input.csv", input_neurons, delimiter=",")
    np.savetxt("labels.csv", input_neurons_labels, delimiter=",")

    # pd.DataFrame(input_neurons).to_csv('/input.csv')
    # pd.DataFrame(input_neurons_labels).to_csv('/labels.csv')

    print("OK")
