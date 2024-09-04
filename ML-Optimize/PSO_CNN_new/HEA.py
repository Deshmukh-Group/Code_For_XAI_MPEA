import itertools
import random
import sys

import numpy as np
import torch
import joblib
import pandas as pd

from CNN_Models import SimpleCNN

device = torch.device('cpu')

model = SimpleCNN()
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

# load Cu_USF_scaler
scaler = joblib.load('Cu_USF_scaler.pkl')
model_usf = joblib.load('model_SEM.pkl')

n1 = float(sys.argv[1]) + 5
n2 = float(sys.argv[2]) + 5
n3 = float(sys.argv[3]) + 5
n4 = float(sys.argv[4]) + 5
n5 = 100-n1-n2-n3-n4


# In[]:

if n5 < 5 or n5 > 35:
    prop = [10000, 10000, 10000, 10000, 10000, 10000]
else:
    # n1, n2, n3, n4 = solution[0], solution[1], solution[2], solution[3]

    N_total = 100

    f1 = n1 / N_total
    f2 = n2 / N_total
    f3 = n3 / N_total
    f4 = n4 / N_total

    natoms = 4000
    v1 = round(f1 * natoms)
    v2 = round(f2 * natoms)
    v3 = round(f3 * natoms)
    v4 = round(f4 * natoms)
    v5 = natoms - (v1 + v2 + v3 + v4)

    list1 = list(itertools.repeat(1, v1))
    list2 = list(itertools.repeat(2, v2))
    list3 = list(itertools.repeat(3, v3))
    list4 = list(itertools.repeat(4, v4))
    list5 = list(itertools.repeat(5, v5))

    list_use = list1 + list2 + list3 + list4 + list5

    if len(list_use) != 4000:
        prop = [10000, 10000, 10000, 10000, 10000, 10000]
    else:
        y_pred = []
        for i in range(10):
            random.shuffle(list_use)
            X = torch.LongTensor(list(list_use))
            X = X.reshape(1, 4000)
            y_temp = model(X)
            y_pred.append(y_temp[0].tolist())
        array_pred = np.array(y_pred)
        prop_avg = np.average(array_pred, axis=0)

        # fitness = 1.0 / (mean_squared_error(y_target, [prop_avg[0]]) + 0.0000001)

        solution_new = [n1, n2, n3, n4, n5]
        # make solution_new to a dataframe
        solution_new = pd.DataFrame(np.reshape(solution_new, (1, 5)), columns=['v1', 'v2', 'v3', 'v4', 'v5'])
        solution_new_scaled = scaler.transform(solution_new)
        prop_usf = model_usf.predict(solution_new_scaled)
        prop = np.append(prop_avg, prop_usf)

with open('a.txt', 'w') as f:
    for item in prop:
        f.write("%s " % item)

