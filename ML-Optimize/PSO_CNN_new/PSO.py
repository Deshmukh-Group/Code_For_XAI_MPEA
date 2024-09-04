import csv
import itertools
import os
import random

from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
import numpy as np
import torch
import joblib
import pandas as pd
from CNN_Models import SimpleCNN
from sklearn.metrics import mean_squared_error

# remove output.csv if it exists
try:
    os.remove("output.csv")
except OSError:
    pass

device = torch.device('cpu')

model = SimpleCNN()
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

# load Cu_USF_scaler
scaler = joblib.load('Cu_USF_scaler.pkl')
model_usf = joblib.load('model_SEM.pkl')

y_target = [210, 270, 180, 140, 160, 425]
#y_target = [210]


# create a parameterized version of the classic Rosenbrock unconstrained optimzation function
def Cu_PSO(x):
    fitness_list = []
    for element in x:
        # print(element)
        n1 = element[0]
        n2 = element[1]
        n3 = element[2]
        n4 = element[3]
        n5 = 100 - n1 - n2 - n3 - n4

        # In[]:

        if n5 < 5 or n5 > 35:
            prop = [10000, 10000, 10000, 10000, 10000, 10000]
            fitness = 1000000
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
                fitness = 1000000
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

                fitness = mean_squared_error(y_target, prop)
                # fitness = mean_squared_error(y_target, [prop[0]])
                output = np.append(solution_new.values.tolist()[0], prop)
                # save the output to a csv file
                with open('output.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(output)

                # print(output)
        # add the fitness to the fitness list
        fitness_list.append(fitness)

    return fitness_list


# instatiate the optimizer
# x_max = 35 * np.ones(4)
# x_min = 5 * np.ones(4)
x_min = np.array([5, 5, 21, 24])
x_max = np.array([9.5, 11, 35, 35])

# make a tuple of a1 and a2 to be used in the bounds
#bounds = tuple((x_min, x_max))
bounds = (x_min, x_max)

options = {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298}
optimizer = GlobalBestPSO(n_particles=128, dimensions=4, options=options, bounds=bounds)

# now run the optimization, pass a=1 and b=100 as a tuple assigned to args

cost, pos = optimizer.optimize(Cu_PSO, iters=600)

pos_history = optimizer.pos_history
cost_history = optimizer.cost_history

# save the cost_history
np.save('cost_history.npy', cost_history)
np.save('pos_history.npy', pos_history)

plot_cost_history(optimizer.cost_history)
plt.show()
