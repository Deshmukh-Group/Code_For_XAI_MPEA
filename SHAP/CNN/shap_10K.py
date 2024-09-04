# coding: utf-8

# In[1]:

import os
import random

import numpy as np
# In[epoch]:
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
#from torchmetrics.functional import r2_score, mean_squared_error
import sys

# from pytorchtools import EarlyStopping

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_data = 'X_train.pt'
y_data = 'y_train.pt'

X_test_data = 'X_test.pt'
y_test_data = 'y_test.pt'

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
        #x = self.embedding(x)
        #x = x.permute(0, 2, 1)
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


model = SimpleCNN()
model.to(device)

# In[24]:


X = torch.load(str(X_data), map_location=torch.device(device))
# map X to cpu
X = X.to(device)

y = torch.load(str(y_data), map_location=torch.device(device))
# map y to cpu
y = y.to(device)

X_test = torch.load(str(X_test_data), map_location=torch.device(device))
# map X to cpu
X_test = X_test.to(device)

y_test = torch.load(str(y_test_data), map_location=torch.device(device))
# map y to cpu
y_test = y_test.to(device)


mean_tmp = torch.mean(y, axis=0)
std_tmp = torch.std(y, axis=0)
print("Column-wise Data Mean:\n", mean_tmp)
print("Column-wise Data Standard deviation:\n", std_tmp)

# load the last checkpoint with the best model
model.load_state_dict(torch.load('./checkpoint.pt', map_location=torch.device(device)))


# use shap to explain the prediction of the first element
import shap
import joblib

# select a set of background examples to take an expectation over
background = X

background_embedded = model.embedding(background).detach()
background_embedded = background_embedded.permute(0, 2, 1)

# explain predictions of the model on four images
e = shap.GradientExplainer(model, background_embedded)

input_data = X_test
input_data_embedded = model.embedding(input_data).detach()
input_data_embedded = input_data_embedded.permute(0, 2, 1)
shap_values = e(input_data_embedded)

joblib.dump(shap_values, 'shap_values.pkl')

# In[]:

for i in np.arange(0, len(shap_values)):
    shap_values_temp = shap_values[i]
    shap_values_temp = shap_values_temp.values
    shap_values_temp_pooled = np.mean(shap_values_temp, axis=1)
    shap_values_averaged = np.mean(shap_values_temp_pooled, axis=0)
    np.save('shap_values_' + str(i) + '.npy', shap_values_averaged)

