# Lets start with the imports.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import time

# Setting number of features and batch size
NUMBER_OF_FEATURES = 11
BATCH_SIZE = 100
WINDOW_SIZE = 10

# Starting counting the time
t0 = time.time()

# We read the dataset and create an iterable.

class my_sensors_NNLL(data.Dataset):
    def __init__(self, input_data, input_data_labels):
        self.data = torch.from_numpy(input_data).float()  # Input data (windows)

        temp_arr = input_data_labels
        final_arr = []

        # Reshaping array to get normalized feature vector
        for x in temp_arr:
            final_arr.append(int(x[0]) - 1)

        final_arr = np.asarray(final_arr)

        unique, counts = np.unique(final_arr, return_counts=True)
        print(dict(zip(unique, counts)))

        self.target = torch.from_numpy(final_arr).float() # Labels for input
        self.n_samples = self.data.shape[0]  # Length of input

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor([self.target[index]])


# We build a simple model with the inputs and one output layer.
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.n_in = 230
        self.n_out = 12

        self.algo = nn.Sequential(
            nn.Linear(self.n_in, 1128),
            nn.ReLU(),
            nn.Linear(1128, 2256),
            nn.ReLU(),
            nn.Linear(2256, 564),
            nn.ReLU(),
            nn.Linear(564, self.n_out)
        )

    def forward(self, x):
        x = self.algo(x)
        return x


# Now, we create the mode, the loss function or criterium and the optimizer
# that we are going to use to minimize the loss.

# Model.
model = MyModel()

# Negative log likelihood loss.
criteria = nn.CrossEntropyLoss()

# Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-9)


# defining Test function
def Test():
    # Test
    print("---- Begin test ----")
    running_loss_test = 0.0
    correct = 0
    total_test = 0
    correct_by_class = [0.01 for x in range(NUMBER_OF_FEATURES + 1)]
    total_by_class = [0.01 for x in range(NUMBER_OF_FEATURES + 1)]
    for l, (sensors_data_test, target_test) in enumerate(my_loader_2):

        # Resizing array for NLLLoss
        final_arr_test = []

        # Reshaping array to get normalized feature vector
        for c in target_test:
            final_arr_test.append(int(c[0]))

        target_test = torch.Tensor(np.asarray(final_arr_test))

        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute the
        # derivative of the inputs.
        sensors_data_test = Variable(sensors_data_test, requires_grad=False)
        target_test = Variable(target_test.long(), requires_grad=False)

        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward.
        pred_test = model(sensors_data_test)

        # Acuurancy counting
        target_indieces = target_test.cpu().data
        prediction_indieces = np.argmax(pred_test.cpu().data, axis=1)

        # Counting  by classes
        for index, data_test in enumerate(target_indieces):
            total_by_class[int(data_test.item())] += 1
            if data_test.item() == prediction_indieces[index]:
                correct_by_class[int(data_test.item())] += 1

        total_test += target_test.size(0)  # total target size
        correct += (target_indieces == prediction_indieces).sum().item()  # counting correct one

        # Loss calculation.
        loss_test = criteria(pred_test, target_test)
        # Gradient calculation.
        loss_test.backward()

        running_loss_test += loss_test.item()

    print(running_loss_test)

    print('Accuracy of the network on the test: %d %%' % (
            100 * correct / total_test))

    print('Accuracy by classes \n')

    for (index, data_total) in enumerate(total_by_class):
        print("Accuracy of class {} - {} %".format(index, (correct_by_class[index] / data_total) * 100))


if __name__ == '__main__':

    print('------ Data loading... ------')

    dataset = pd.read_csv('zero_input.csv', delimiter=",", header=None, dtype=np.float32).values  # Read data file.
    dataset_labels = pd.read_csv('zero_labels.csv', delimiter=",", header=None,
                                 dtype=np.float32).values  # Read data file.

    tr_data = dataset[:int(dataset.shape[0] * 0.7)]
    tr_data_labels = dataset_labels[:int(dataset_labels.shape[0] * 0.7)]

    test_data = dataset[int(dataset.shape[0] * 0.7):]
    test_data_labels = dataset_labels[int(dataset.shape[0] * 0.7):]

    training_data = my_sensors_NNLL(tr_data, tr_data_labels)
    test_data = my_sensors_NNLL(test_data, test_data_labels)

    # We create the dataloader for both data
    my_loader = data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    my_loader_2 = data.DataLoader(test_data, batch_size=BATCH_SIZE)

    print('------ Data loading end ------')

    # Training
    print("----- Begin training -----")

    for epoch in range(3):

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

        print('\n')
        print('Test %d', epoch)
        print(epoch)
        Test()
        print(running_loss)

    Test()
    t1 = time.time()
    total = t1 - t0
    print("TIME ELAPSED: {} seconds OR {} minutes".format(total, total / 60.0))

