import itertools
import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

import pygad
from CNN_Models import SimpleCNN


# In[]
device = torch.device('cpu')

model = SimpleCNN()
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

y_target = [200]

def fitness_func(solution, solution_idx):
    #print(solution)

    list_use = list(map(int, solution))

    X = torch.LongTensor(list(list_use))
    X = X.reshape(1, 4000)
    y_temp = model(X)
    y_pred = y_temp[0].tolist()

    fitness = 1.0 / (mean_squared_error(y_target, [y_pred[0]]) + 0.0000001)
    return fitness


# In[]

num_generations = 10 # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
mutation_percent_genes = 10
sol_per_pop = 10  # Number of solutions in the population.
num_genes = 4000  # Number of atoms

SPACE = np.arange(1, 6, 1)
gene_space = [SPACE]*4000
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
                       on_generation=on_generation)

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
