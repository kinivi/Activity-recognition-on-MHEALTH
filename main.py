# Lets start with the imports.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time

# Setting number of features
NUMBER_OF_FEATURES = 12
BATCH_SIZE = 500

# Starting counting the time
t0 = time.time()

# Create testset and dataset
dataset = pd.read_csv('subject_1.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
li = dataset
dataset = pd.read_csv('subject_2.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.
li = np.append(li, dataset, axis=0)
dataset = pd.read_csv('subject_3.txt', delimiter=" ", header=None, dtype=np.float32).values
dataset = np.append(li, dataset, axis=0)


tr_data = dataset[:int(dataset.shape[0] * 0.75)]
test_data = dataset[int(dataset.shape[0] * 0.70):]


# We read the dataset and create an iterable.
class my_sensors(data.Dataset):
    def __init__(self, input_data):
        self.data = input_data[:, :23]  # 1st and 2nd columns --> x,y

        temp_arr = input_data[:, 23:]
        final_arr = []

        # Reshaping array to get normalized feature vector
        for x in temp_arr:
            number_of_features = NUMBER_OF_FEATURES

            temp_a = [0. for x in range(number_of_features + 1)]
            temp_a[int(x)] = 1.
            final_arr.append(temp_a)

        final_arr = np.asarray(final_arr)
        self.target = final_arr
        self.n_samples = self.data.shape[0]

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])


class my_sensors_NNLL(data.Dataset):
    def __init__(self, input_data):
        self.data = input_data[:, :23]  # 1st and 2nd columns --> x,y

        temp_arr = input_data[:, 23:]
        final_arr = []

        # Reshaping array to get normalized feature vector
        for x in temp_arr:
            final_arr.append(int(x[0]))

        self.target = np.asarray(final_arr)
        print(self.target[0])
        self.n_samples = self.data.shape[0]


    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        #print(torch.Tensor([self.target[index]]))
        return torch.Tensor(self.data[index]), torch.Tensor([self.target[index]])


training_data = my_sensors_NNLL(tr_data)
test_data = my_sensors_NNLL(test_data)


# We create the dataloader for both data
my_loader = data.DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=2)
my_loader_2 = data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)


# We build a simple model with the inputs and one output layer.
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        self.n_in = 23
        self.n_out = 13

        self.algo = nn.Sequential(
            nn.Linear(self.n_in, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_out),

        )

    def forward(self, x):
        x = self.algo(x)
        return x


# Now, we create the mode, the loss function or criterium and the optimizer
# that we are going to use to minimize the loss.

# Model.
model = my_model()

# Negative log likelihood loss.
criteria = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# Training
print("----- Begin training -----")

for epoch in range(14):

    running_loss = 0.0
    for k, (sensors_data, target) in enumerate(my_loader):

        # Resizing array for NLLLoss
        final_arr = []

        # Reshaping array to get normalized feature vector
        for x in target:
            final_arr.append(int(x[0]))

        target = torch.Tensor(np.asarray(final_arr))

        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute the
        # derivative of the inputs.
        sensors_data = Variable(sensors_data, requires_grad=True)
        target = Variable(target.long(), requires_grad=False)
        target_NLLL = Variable(torch.FloatTensor(500).uniform_(0, 12).long())

        # Set gradient to 0
        optimizer.zero_grad()
        # Feed forward.
        pred = model(sensors_data)
        # Loss calculation.

        loss = criteria(pred, target)
        # Gradient calculation.
        loss.backward()

        running_loss += loss.item()
        # Print loss every 10 iterations.
        if k % 1000 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, k + 1, loss))

            # print(target, pred)

        # Model weight modification based on the optimizer.
        optimizer.step()

    print(running_loss)




















# Test
print("---- Begin test ----")

running_loss = 0.0
correct = 0
total = 0
for k, (sensors_data, target) in enumerate(my_loader_2):

    # Resizing array for NLLLoss
    final_arr = []

    # Reshaping array to get normalized feature vector
    for x in target:
        final_arr.append(int(x[0]))

    target = torch.Tensor(np.asarray(final_arr))

    # Definition of inputs as variables for the net.
    # requires_grad is set False because we do not need to compute the
    # derivative of the inputs.
    sensors_data = Variable(sensors_data, requires_grad=False)
    target = Variable(target.long(), requires_grad=False)
    target_NLLL = Variable(torch.FloatTensor(500).uniform_(0, 12).long())

    # Set gradient to 0.
    optimizer.zero_grad()
    # Feed forward.
    pred = model(sensors_data)

    #Acuurancy counting
    target_indieces = target.data
    prediction_indieces = np.argmax(pred.data, axis=1)

    total += target.size(0)

    correct += (target_indieces == prediction_indieces).sum().item()

    # Loss calculation.
    loss = criteria(pred, target)
    # Gradient calculation.
    loss.backward()

    running_loss += loss.item()


print(running_loss)

print('Accuracy of the network on the test: %d %%' % (
    100 * correct / total))

t1 = time.time()
total = t1-t0
print("TIME ELAPSED: {} seconds OR {} minutes".format(total, total/60.0))
