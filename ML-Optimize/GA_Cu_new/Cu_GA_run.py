import csv
import itertools
import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

import pygad
from CNN_Models import SimpleCNN

# remove output.csv if it exists
try:
    os.remove("output.csv")
except OSError:
    pass


# In[]
device = torch.device('cpu')

model = SimpleCNN()
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

# load Cu_USF_scaler
scaler = joblib.load('Cu_USF_scaler.pkl')
model_usf = joblib.load('model_SEM.pkl')

# In[]

y_target = [210, 270, 180, 140, 160, 425]


# y_target = [183.3668, 244.7649, 152.6678, 114.2439, 127.4657] # 9.65549737  9.1516309  27.49052619 25.58225388  9.5611476
# y_target = [200]  # 5.16159303  5.19397743 33.80025292 26.78998839 // fitness 0.008711602116755228


def fitness_func(solution, solution_idx):
    n1, n2, n3, n4 = solution[0], solution[1], solution[2], solution[3]
    n5 = 100 - (n1 + n2 + n3 + n4)
    if n5 < 5 or n5 > 35:
        fitness = 1e-9
        #prop_avg = -10000
        #prop_usf = -10000
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
            fitness = 1e-9
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

            fitness = 1.0 / (mean_squared_error(y_target, prop) + 0.0000001)
            output = np.append(solution_new.values.tolist()[0], prop)
            # save the output to a csv file
            with open('output.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(output)

            #print(output)
    # print(solution, prop_avg, prop_usf)
    return fitness



# In[]

num_generations = 600  # Number of generations.
num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.
mutation_percent_genes = 50
sol_per_pop = 128  # Number of solutions in the population.
num_genes = 4  # Number of atoms

# SPACE = np.arange(1,6,1)
# gene_space = [SPACE]*4000


SPACE1 = {'low': 5, 'high': 9.5}
SPACE2 = {'low': 5, 'high': 11}
SPACE3 = {'low': 21, 'high': 35}
SPACE4 = {'low': 24, 'high': 35}
gene_space = [SPACE1, SPACE2, SPACE3, SPACE4]
last_fitness = 0


def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(
        change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))

    solution1, solution_fitness1, solution_idx1 = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution1))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness1))

    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=gene_space,
                       on_generation=on_generation,
                       #save_solutions=True,
                       #allow_duplicate_genes=False
                       )

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

# save ga_instance.best_solutions_fitness to a npy file
np.save('best_solution_fitness.npy', ga_instance.best_solutions_fitness)