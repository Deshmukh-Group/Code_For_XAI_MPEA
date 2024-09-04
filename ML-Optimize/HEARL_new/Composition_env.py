import csv
import itertools
import time
import random
import numpy as np
import os
import sys

import torch
import joblib
import pandas as pd
from CNN_Models import SimpleCNN
from sklearn.metrics import mean_squared_error


class CompositionEnv(object):
    # define actions

    def __init__(self, start_config=None, goal_config=None, max_steps=None, ):
        # define the initial composition

        self.score = 0
        self.initial_states = start_config
        self.goal = goal_config

        self.y_target = [210, 270, 180, 140, 160, 425]
        self.is_reset = False
        self.ACTIONS_DICT = {}
        self.state = start_config
        self.num = 0
        self.max_steps = max_steps

        k = 0
        for i in range(len(self.initial_states)):
            for j in [-1, 1]:
                # print(k, i, j)
                # add k: (i, j) to ACTIONS_DICT
                self.ACTIONS_DICT[k] = (i, j)
                k += 1

        self.device = torch.device('cpu')

        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('checkpoint.pt', map_location=self.device))

        # load Cu_USF_scaler
        self.scaler = joblib.load('Cu_USF_scaler.pkl')
        self.model_usf = joblib.load('model_SEM.pkl')

    def evaluation(self, state):
        n1 = self.state[0]
        n2 = self.state[1]
        n3 = self.state[2]
        n4 = self.state[3]
        n5 = 100 - n1 - n2 - n3 - n4

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
                    y_temp = self.model(X)
                    y_pred.append(y_temp[0].tolist())
                array_pred = np.array(y_pred)
                prop_avg = np.average(array_pred, axis=0)

                # fitness = 1.0 / (mean_squared_error(y_target, [prop_avg[0]]) + 0.0000001)

                solution_new = [n1, n2, n3, n4, n5]
                # make solution_new to a dataframe
                solution_new = pd.DataFrame(np.reshape(solution_new, (1, 5)), columns=['v1', 'v2', 'v3', 'v4', 'v5'])
                solution_new_scaled = self.scaler.transform(solution_new)
                prop_usf = self.model_usf.predict(solution_new_scaled)
                prop = np.append(prop_avg, prop_usf)

                fitness = mean_squared_error(self.y_target, prop)
                output = np.append(solution_new.values.tolist()[0], prop)
                # save the output to a csv file
                with open('output.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(output)

        return fitness

    # create a parameterized version of the classic Rosenbrock unconstrained optimzation function

    def step(self, action: int):
        """
        Takes a given action in the environment's current state, and returns a next state,
        reward, and whether the next state is terminal or not.

        Arguments:
            action {int} -- The action to take in the environment's current state. Should be an integer in the range [0-7].

        Raises:
            RuntimeError: Raised when the environment needs resetting.\n
            TypeError: Raised when an action of an invalid type is given.\n
            ValueError: Raised when an action outside the range [0-7] is given.\n

        Returns:
            A tuple of:\n
                {(float, float, float, float)} -- The next state (new composition, a tuple of (c1,c2,c3,c4).\n
                {float} -- The reward earned by taking the given action in the current environment state.\n
                {bool} -- Whether the environment's next state is terminal or not.\n

        """

        # Check whether a reset is needed.
        if (not self.is_reset):
            raise RuntimeError(".step() has been called when .reset() is needed.\n" +
                               "You need to call .reset() before using .step() for the first time, and after an episode ends.\n" +
                               ".reset() initialises the environment at the start of an episode, then returns an initial state.")

        # Check that action is the correct type (either a python integer or a numpy integer).
        if (not (isinstance(action, int) or isinstance(action, np.integer))):
            raise TypeError("action should be an integer.\n" +
                            "action value {} of type {} was supplied.".format(action, type(action)))

        # Check that action is an allowed value.
        if (action < 0 or action > 7):
            raise ValueError(
                "action must be an integer in the range [0-7] corresponding to one of the legal actions.\n" +
                "action value {} was supplied.".format(action))

        # Update State
        (c_loc, c_change) = self.ACTIONS_DICT[action]
        self.state = list(self.state)
        self.state[c_loc] += c_change * abs(np.random.normal(0, 1))

        # Check that composition is within bounds.
        for i in range(len(self.state)):
            if (self.state[i] > 35):
                self.state[i] = 35
            elif (self.state[i] < 5):
                self.state[i] = 5

        # Initial reward
        new_composition = self.state
        last_element = 100 - sum(new_composition)
        reward = 0
        terminal = False

        # value function
        value = self.evaluation(state=new_composition)
        self.score = abs(value - self.goal)
        # print(self.score)
        # (self.num)

        # If last element is not within 5 to 35, then the composition is not balanced.
        if (last_element < 5 or last_element > 35):
            reward -= 100000

        if self.score > 0.001 * self.goal:
            reward -= self.score
        else:
            reward += 1000000
            # print(self.state)
            terminal = True

        if self.num == self.max_steps:
            terminal = True
            # print("terminal")
            self.num = 0
        else:
            self.num += 1

        # Penalise every timestep.
        reward -= 100

        # Require a reset if the current state is terminal.
        if terminal:
            self.is_reset = False

        self.state = tuple(self.state)
        # Return next state, reward, and whether the episode has ended.
        return (self.state[0], self.state[1], self.state[2], self.state[3]), reward, terminal

    def reset(self):
        """
        Resets the environment, ready for a new episode to begin, then returns an initial state.
        The initial state will be a starting grid square randomly chosen using a uniform distribution,
        with both components of the velocity being zero.

        Returns:
            {(float,float,float,float)} -- an initial state, a tuple of (c1,c2,c3,c4).
        """

        self.state = self.initial_states
        self.is_reset = True

        return self.state

    def get_actions(self):
        """
        Returns the available actions in the current state - will always be a list
        of integers in the range [0-7].
        """
        return [*self.ACTIONS_DICT]






