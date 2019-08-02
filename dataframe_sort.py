# Lets start with the imports.
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time
from sklearn.utils import shuffle


print('------ Data loading... ------')
# Create testset
dataset = pd.read_csv('features_test_encoded16_ds1_batch_sig.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.

dataframe = pd.DataFrame(data = dataset)
dataframe = dataframe.sort_values(by=16, axis=0)
print(dataframe)
dataframe.to_csv(r'export_dataframe.csv', index = None, header=None)