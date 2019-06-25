# Lets start with the imports.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time

#  Setting CUDA
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Setting number of features and batch size
NUMBER_OF_FEATURES = 11
BATCH_SIZE = 5
WINDOW_SIZE = 10

# Starting counting the time
t0 = time.time()


print('------ Data loading... ------')

dataset = pd.read_csv('zero_input.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
dataset_labels = pd.read_csv('zero_labels.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.

tr_data = dataset[:int(dataset.shape[0] * 0.6)]
tr_data_labels = dataset_labels[:int(dataset_labels.shape[0] * 0.6)]

test_data = dataset[int(dataset.shape[0] * 0.6):]
test_data_labels = dataset_labels[int(dataset.shape[0] * 0.6):]




# We read the dataset and create an iterable.
class PlaneDataset(data.Dataset):
    def __init__(self, input_data, input_data_labels):

        self.data = torch.from_numpy(input_data).float().cuda()  # Input data (windows)

        temp_arr = input_data_labels
        final_arr = []

        # Reshaping array to get normalized feature vector
        for x in temp_arr:
            number_of_features = NUMBER_OF_FEATURES

            temp_a = [0. for x in range(number_of_features + 1)]
            temp_a[int(x)-1] = 1.
            final_arr.append(temp_a)

        final_arr = np.asarray(final_arr)
        self.target = torch.from_numpy(final_arr).float().cuda()  # Labels for input

        self.n_samples = self.data.shape[0]  # Length of input

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])


training_data = PlaneDataset(tr_data, tr_data_labels)
test_data = PlaneDataset(test_data, test_data_labels)

# We create the dataloader for both data
my_loader = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
my_loader_2 = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print('------ Data loading end ------')


# ------------------- Model section ----------------------

# We build a simple model with the inputs and one output layer.

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.n_in = 230
        self.n_out = 12

        self.algo = nn.Sequential(
            nn.Linear(self.n_in, 256),
            nn.Linear(256, 564),
            nn.Linear(564, 256),
            nn.Linear(256, 64),
            nn.Linear(64, self.n_out),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.algo(x)
        return x


# Now, we create the mode, the loss function or criterium and the optimizer
# that we are going to use to minimize the loss.

# Model.
model = MyModel()
MyModel.cuda(model, device)

# Negative log likelihood loss.
criteria = nn.MSELoss()

# Adam optimizer with learning rate 0.01 and L2 regularization with weight 1e-4.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------- Model section ----------------------

# ------------------- Training section ----------------------
# Training
print("----- Begin training -----")

for epoch in range(3):

    running_loss = 0.0
    for k, (sensors_data, target) in enumerate(my_loader):

        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute the
        # derivative of the inputs.
        sensors_data_v = Variable(sensors_data, requires_grad=False).cuda(device)
        target_v = Variable(target, requires_grad=False).cuda(device)

        # Set gradient to 0
        optimizer.zero_grad()
        # Feed forward.
        pred = model(sensors_data_v)
        # Loss calculation.

        loss = criteria(pred, target_v)
        # Gradient calculation.
        loss.backward()

        running_loss += loss.item()
        # Print loss every 10 iterations.
        if k % 550 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, k + 1, loss))

            # print(target, pred)

        # Model weight modification based on the optimizer.
        optimizer.step()

    print(running_loss)

# ------------------- Training section ----------------------


# ---------------------- TEST PART -------------------------

# Test
print("---- Begin test ----")

running_loss = 0.0
correct = 0
total = 0
correct_by_class = [0.01 for x in range(NUMBER_OF_FEATURES + 1)]
total_by_class = [0.01 for x in range(NUMBER_OF_FEATURES + 1)]

for k, (sensors_data, target) in enumerate(my_loader_2):

    # Definition of inputs as variables for the net.
    # requires_grad is set False because we do not need to compute the
    # derivative of the inputs.
    sensors_data = Variable(sensors_data, requires_grad=False).cuda(device)
    target = Variable(target, requires_grad=False).cuda(device)

    # Set gradient to 0.
    optimizer.zero_grad()
    # Feed forward.
    pred = model(sensors_data)

    # Acuurancy counting
    target_indieces = np.argmax(target.cpu().data, axis=1)
    prediction_indieces = np.argmax(pred.cpu().data, axis=1)

    # Counting  by classes
    for index, data in enumerate(target_indieces):
        total_by_class[int(data.item())] += 1
        if data.item() == prediction_indieces[index]:
            correct_by_class[int(data.item())] += 1

    total += target.size(0)  # total target size
    correct += (target_indieces == prediction_indieces).sum().item()  # counting correct one

    # Loss calculation.
    loss = criteria(pred, target)
    # Gradient calculation.
    loss.backward()

    running_loss += loss.item()


print(running_loss)

print('Accuracy of the network on the test: %d %%' % (
        100 * correct / total))

print('Accuracy by classes \n')

for (index, data) in enumerate(total_by_class):
    print("Accuracy of class {} - {} %".format(index, (correct_by_class[index] / data) * 100))

t1 = time.time()
total = t1 - t0
print("TIME ELAPSED: {} seconds OR {} minutes".format(total, total / 60.0))
