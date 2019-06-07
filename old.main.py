# Lets start with the imports.
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

# First we create the point that we are going to use for the classifier.
# We create n_points points for four classes of points center at [0,0],
# [0,2], [2,0] and [2,2] with a deviation from the center that follows a
# Gaussian distribution with a standar deviation of sigma.

# n_points = 20000
# points = np.zeros((n_points,2))   # x, y
# target = np.zeros((n_points,1))   # label
# sigma = 0.5
# for k in range(n_points):
#     # Random selection of one class with 25% of probability per class.
#     random = np.random.rand()
#     if random<0.25:
#         center = np.array([0,0])
#         target[k,0] = 0   # This points are labeled 0.
#     elif random<0.5:
#         center = np.array([2,2])
#         target[k,0] = 1   # This points are labeled 1.
#     elif random<0.75:
#         center = np.array([2,0])
#         target[k,0] = 2   # This points are labeled 2.
#     else:
#         center = np.array([0,2])
#         target[k,0] = 3   # This points are labeled 3.
#     gaussian01_2d = np.random.randn(1,2)
#     points[k,:] = center + sigma*gaussian01_2d
#
# # Now, we write all the points in a file.
# points_and_labels = np.concatenate((points,target),axis=1)   # 1st, 2nd, 3nd column --> x,y, label
# pd.DataFrame(points_and_labels).to_csv('clas.csv',index=False)

# Setting number of features
NUMBER_OF_FEATURES = 12

# Create testset and dataset
dataset = pd.read_csv('subject_1.txt', delimiter=" ", header=None, dtype=np.float32).values  # Read data file.

tr_data = dataset[:int(dataset.shape[0] * 0.80)]
test_data = dataset[int(dataset.shape[0]*0.80):]


# We read the dataset and create an iterable.
class my_sensors(data.Dataset):
    def __init__(self, input_data):

        self.data = input_data[:, :23]  # 1st and 2nd columns --> x,y

        temp_arr = input_data[:, 23:]
        final_arr = []

        for x in temp_arr:

            number_of_features = NUMBER_OF_FEATURES

            temp_a = [0. for x in range(number_of_features + 1)]
            temp_a[int(x)] = 1.
            final_arr.append(temp_a)

            # temp_arr = [0 for z in range(13)]
            # max_x = self.target[int(x)].max()
            #
            # temp_arr[int(max_x)] = 1
            #
            # #self.target[int(x)].reshape((-1, 13))
            # self.target[int(x)] = np.asarray(temp_arr)

        final_arr = np.asarray(final_arr)
        self.target = final_arr

       # temp_arr = [[if x[0] ==5 z else 0 for z in range(13)] for x in self.target]
        #temp_arr = np.reshape(self.target, (13, -1))

        #temp_arr = np.asarray(temp_arr)

        self.n_samples = self.data.shape[0]

    def __len__(self):  # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):  # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])


# We create the dataloader.
training_data = my_sensors(tr_data)
test_data = my_sensors(test_data)
batch_size = 50
my_loader = data.DataLoader(training_data, batch_size=batch_size, num_workers=2)
my_loader_2 = data.DataLoader(test_data, batch_size=batch_size, num_workers=2)


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
criterium = nn.MSELoss()

# Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training.
for epoch in range(1):

    running_loss = 0.0
    for k, (sensors_data, target) in enumerate(my_loader):

        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute the
        # derivative of the inputs.
        sensors_data = Variable(sensors_data, requires_grad=False)
        target = Variable(target.float(), requires_grad=False)

        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward.
        pred = model(sensors_data)
        # Loss calculation.

        loss = criterium(pred, target)
        # Gradient calculation.
        loss.backward()

        running_loss += loss.item()
        # Print loss every 10 iterations.
        if k % 1000 == 0:
            print('[%d, %5d] loss: %.8f' %
                  (epoch + 1, k + 1, loss))
            running_loss = 0.0
            #print(target, pred)

        # Model weight modification based on the optimizer.
        optimizer.step()


for k, (sensors_data, target) in enumerate(my_loader_2):

    # Definition of inputs as variables for the net.
    # requires_grad is set False because we do not need to compute the
    # derivative of the inputs.
    sensors_data = Variable(sensors_data, requires_grad=False)
    target = Variable(target.float(), requires_grad=False)

    # Set gradient to 0.
    optimizer.zero_grad()
    # Feed forward.
    pred = model(sensors_data)
    # Loss calculation.

    loss = criterium(pred, target)
    # Gradient calculation.
    loss.backward()

    running_loss += loss.item()
    # Print loss every 10 iterations.
    if k % 1000 == 0:
        print('[%d, %5d] loss: %.8f' %
              (epoch + 1, k + 1, loss))
        running_loss = 0.0
        # print(target, pred)

#
#
#
#
# # Now, we plot the results.
# # Circles indicate the ground truth and the squares are the predictions.
#
# colors = ['r','b','g','y']
# points = data.numpy()
#
# # Ground truth.
# target = target.numpy()
# for k in range(4):
#     select = target[:,0]==k
#     p = points[select,:]
#     plt.scatter(p[:,0],p[:,1],facecolors=colors[k])
#
# # Predictions.
# pred = pred.exp().detach()     # exp of the log prob = probability.
# _, index = torch.max(pred,1)   # index of the class with maximum probability.
# pred = pred.numpy()
# index = index.numpy()
# for k in range(4):
#     select = index==k
#     p = points[select,:]
#     plt.scatter(p[:,0],p[:,1],s=60,marker='s',edgecolors=colors[k],facecolors='none')
#
# plt.show()