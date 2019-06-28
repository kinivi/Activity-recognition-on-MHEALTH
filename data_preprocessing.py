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
from sklearn.utils import shuffle


WINDOWS_SIZE = 10
WINDOWS_STEP = 1

print('------ Data loading... ------')
# Create testset andnn.Linear(128, 256), dataset
dataset = pd.read_csv('subject_1.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
li = dataset
# dataset = pd.read_csv('subject_2.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_3.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_4.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_5.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_6.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_7.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_8.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_9.txt', delimiter=" ", header=None, dtype=np.float32).values
# li = np.append(li, dataset, axis=0)
# dataset = pd.read_csv('subject_10.txt', delimiter=" ", header=None, dtype=np.float32).values
dataset = np.append(li, dataset, axis=0)

# Shuffling dataset
# np.random.shuffle(dataset)

# Inint scaler for Normalization
min_max_scaler = preprocessing.MinMaxScaler()

time_part_1 = 0
time_part_2 = 0

if __name__ == '__main__':
    labels = dataset[:, 23:]
    inputs = min_max_scaler.fit_transform(dataset[:, :23])  # Normalization
    input_neurons = []
    input_neurons_labels = []

    # Deleting zeros for optimization
    zero_indieces = []

    for index, (label) in enumerate(labels):
        label = int(label[0])

        if label == 0:
            zero_indieces.append(index)

    inputs = np.delete(inputs, zero_indieces, 0)
    labels = np.delete(labels, zero_indieces, 0)

    print("---------- Zeros deleted ----------")

    for i in range(0, inputs.shape[0], WINDOWS_STEP):

        # Starting counting the time
        t0 = time.time()

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
                input_neurons.append(buffer[0].tolist())
                input_neurons_labels.append(int(window_label[0]))
            except:
                print(buffer.shape)

        if i % 50000 == 0:
            print("Processing...")


    t1 = time.time()
    time_part_1 += t1 - t0

    # Shuffling after creating windows
    input_neurons, input_neurons_labels = shuffle(input_neurons, input_neurons_labels, random_state=0)

    # To numpy array
    input_neurons = np.asarray(input_neurons)
    input_neurons_labels = np.asarray(input_neurons_labels)

    print(input_neurons.shape)
    np.savetxt("step_1. 2csv", input_neurons, delimiter=",")
    np.savetxt("step_2.csv", input_neurons_labels, delimiter=",")

    t2 = time.time()
    time_part_2 += t2 - t1

    print("TIME ELAPSED PART_1: {} seconds OR {} minutes".format(time_part_1, time_part_1 / 60.0))
    print("TIME ELAPSED PART_2: {} seconds OR {} minutes".format(time_part_2, time_part_2 / 60.0))

    print("OK")
