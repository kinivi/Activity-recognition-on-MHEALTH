from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras, device
from time import time
from tensorflow.compat.v1.keras import datasets, layers, models
from tensorflow.python.keras.callbacks import TensorBoard

# Helper libraries
import numpy as np



if __name__ == '__main__':

    print('------ Data loading... ------')

    dataset = pd.read_csv('step_1C.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
    dataset_labels = pd.read_csv('step_2C.csv', delimiter=",", header=None,
                                 dtype=np.float32).values  # Read data file.

    with device('/gpu:1'):
        # Shaping data for CNN
        tr_data = dataset[:int(dataset.shape[0] * 0.75)]
        tr_data = tr_data.reshape(257382, 20, 7)
        tr_data = np.expand_dims(tr_data, axis=3)
        print(tr_data.shape)

        # Reshape labels for appropriate format
        tr_data_labels = keras.utils.to_categorical(dataset_labels[:int(dataset_labels.shape[0] * 0.75)])

        test_data = dataset[int(dataset.shape[0] * 0.75):]
        test_data = test_data.reshape(85794, 20, 7)
        test_data = np.expand_dims(test_data, axis=3)
        print(test_data.shape)
        test_data_labels = keras.utils.to_categorical(dataset_labels[int(dataset.shape[0] * 0.75):])

        # Init statemnts
        print(tf.__version__)

        # Define model

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(20, 7, 1)))
        model.add(layers.Conv2D(32, (3, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(64, (2, 1), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(128, (2, 1), activation='relu'))
        model.add(layers.Flatten())
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(12, activation='softmax'))

        # Init TensorBoard callback for data visualization in real time
        tensorboard = TensorBoard(log_dir="logs/test".format(time()))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=["accuracy", keras.metrics.FalsePositives(), keras.metrics.FalseNegatives(),
                               keras.metrics.Recall(), keras.metrics.Precision()])

        model.fit(tr_data, tr_data_labels, validation_data=(test_data, test_data_labels), epochs=27, callbacks=[tensorboard])

