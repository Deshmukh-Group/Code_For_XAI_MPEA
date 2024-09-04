# coding: utf-8

# In[1]:

import os
import random

import numpy as np
# In[epoch]:
import torch
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score, mean_squared_error

if torch.cuda.is_available():
    device = "cuda:0"
    print("GPU used")
else:
    device = "cpu"
    print("CPU used")

# In[CNN]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()
        self.layer = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        self.pooling = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(984, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 5)
        self.embedding = nn.Embedding(6, 8)

    # @torchsnooper.snoop()
    def forward(self, x, **kwargs):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x


# In[6]

model = SimpleCNN()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

batch_size = 30

# In[24]:

X_train = torch.load('X_train.pt')
X_test = torch.load('X_test.pt')
X_val = torch.load('X_val.pt')

y_train = torch.load('y_train.pt')
y_test = torch.load('y_test.pt')
y_val = torch.load('y_val.pt')

# data to device
X_train = X_train.to(device)
X_test = X_test.to(device)
X_val = X_val.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
y_val = y_val.to(device)

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
train_data = DataLoader(dataset_train, batch_size, shuffle=True)

dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
val_data = DataLoader(dataset_val, batch_size, shuffle=True)

dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
test_data = DataLoader(dataset_test, batch_size, shuffle=True)

# In[Save data]

mean_tmp = torch.mean(y_train, axis=0)
std_tmp = torch.std(y_train, axis=0)
print("Column-wise Train Data Mean:\n", mean_tmp)
print("Column-wise Train Data Standard deviation:\n", std_tmp)
print(torch.max(y_train))

mean_tmp = torch.mean(y_val, axis=0)
std_tmp = torch.std(y_val, axis=0)
print("Column-wise Valid Data Mean:\n", mean_tmp)
print("Column-wise Valid Data Standard deviation:\n", std_tmp)
print(torch.max(y_val))

mean_tmp = torch.mean(y_test, axis=0)
std_tmp = torch.std(y_test, axis=0)
print("Column-wise Test Data Mean:\n", mean_tmp)
print("Column-wise Test Data Standard deviation:\n", std_tmp)
print(torch.max(y_test))

# In[25]:

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

n_epochs = 50000
patience = 2000
early_stopping = EarlyStopping(patience=patience, verbose=True)

# with torchsnooper.snoop():
for epoch in range(1, n_epochs + 1):
    for x, y in train_data:
        yhat = model(x)
        # loss = torch.mean(torch.sum(torch.pow(yhat - y, 2), dim=1))
        loss = criterion(yhat, y)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    for x, y in val_data:
        yhat = model(x)
        loss = criterion(yhat, y)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    early_stopping(valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

# In[ Testing ]

y_pred = model(X_test)
print(r2_score(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))
