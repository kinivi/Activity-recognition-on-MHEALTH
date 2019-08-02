from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import pathlib
import random

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    print('------ Data loading... ------')

    dataset = pd.read_csv('step_1.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
    dataset_labels = pd.read_csv('step_2.csv', delimiter=",", header=None,
                                 dtype=np.float32).values  # Read data file.

    # Shaping data for CNN
    tr_data = dataset[:int(dataset.shape[0] * 0.75)]
    tr_data = tr_data.reshape(257382, 20, 23)
    tr_data = np.expand_dims(tr_data, axis=3)
    print(tr_data.shape)
    tr_data_labels = dataset_labels[:int(dataset_labels.shape[0] * 0.75)]

    test_data = dataset[int(dataset.shape[0] * 0.75):]
    test_data = test_data.reshape(85794, 20, 23)
    test_data = np.expand_dims(test_data, axis=3)
    print(test_data.shape)
    test_data_labels = dataset_labels[int(dataset.shape[0] * 0.75):]

    # Init statemnts
    print(tf.__version__)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 23, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(50, (2, 2), activation='relu'))
    model.add(layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(layers.Conv2D(256, (2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(12, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(tr_data, tr_data_labels, epochs=4)

    test_loss, test_acc = model.evaluate(test_data, test_data_labels)
