# pwpso.py
# python 3.4.3
# searching parameters for polarized water model using particle swarm optimization (PSO)
# runs namd simulations
from __future__ import print_function

import copy  # array-copying convenience
import json
import math  #
import os
import random
import subprocess
import sys  # max float
from typing import Union

DATAFILE = 'datafile.dat'
LOGFILE = 'output.log'
BESTFILE = 'bestresults.dat'
CONFIG_FILE_PATH = "config.json"

swarm = []  # create an empty swarm
epoch = 0


# w = 0.729  # inertia or under-relaxation I guess
# c1 = 1.49445  # cognitive (bird)
# c2 = 1.49445  # social (swarm)


def get_config(cfg_file=CONFIG_FILE_PATH):
    with open(cfg_file) as f:
        return json.load(f)


def show_vector(vector):
    log_file = open(LOGFILE, "a")
    for i in range(len(vector)):
        if i % 8 == 0:  # 8 columns
            print("\n", end="")
        if vector[i] >= 0.0:
            print(' ', end="")
        print("%.4f" % vector[i], end="")  # 4 decimals
        log_file.write("%.4f " % vector[i])
        print(" ", end="")
    print("\n")
    log_file.write("\n")
    log_file.close()


def runsimulation(config):  # runs once per step/eopch
    var = [[0.0 for x in range(config["dim"])] for y in range(config["num_birds"])]
    np = len(swarm)  # number of birds = len(swarm)
    global epoch

    for i in range(np):
        var[i] = swarm[i].var[:]
        #print(swarm[i].var[:])

        os.chdir(str(i))
        file = 'new_par.txt'
        text_file = open(file, "w")  # open("new_par"+i+".txt", "w")
        txt = " ".join(str(var[i][x]) for x in range(config["dim"]))  ######################################## change
        text_file.write(txt)
        text_file.close()
        os.chdir("..")
        if epoch >= config["ann_start_epoch"] and i == config["ann_bird_id"] - 1:
            runann(config)
        command = ['./replace_values.sh %s' % (str(i))]
        p = subprocess.Popen(command, bufsize=2048, shell=True, stdin=subprocess.PIPE)
        p.wait()
        
        command = ['./save_initial_structure.sh %s %s' % (str(i), str(epoch))]
        p = subprocess.Popen(command, bufsize=2048, shell=True, stdin=subprocess.PIPE)
        p.wait()        

    # run ANN here to change the parameters before running the simulations

    print("EPOCH ID" + str(epoch))
    command = ['./PSO_bash.sh %s' % (str(config["num_birds"]))]
    # command=['model1.exe','%d'%(i,)]
    p = subprocess.Popen(command, bufsize=2048, shell=True, stdin=subprocess.PIPE)
    p.wait()
    p.stdin.close()
    return var


def readresults(id, config, var):  # reads one output file at a time from dir id
    print("Reading results from folder " + str(id))
    log_file = open(LOGFILE, "a")
    log_file.write("Reading results from folder " + str(id) + "\n")
    os.chdir(str(id))
    text_file = open("result.txt", "r")
    lines = text_file.read().split()
    text_file.close()
    os.chdir("..")
    pass
    error = 0.0
    data_file = open(DATAFILE, "a")
    for i in range(config["dim"]):
        data_file.write("%f " % (var[id][i]))
    for i in range(0, config["nresults"]):
        try:
            xfloat = float(lines[i])
        except ValueError:
            xfloat = 100.0
        xi = (xfloat - config["target"][i]) / config["target"][i]
        err = math.sqrt((xi * xi)) * config["wt"][i]
        error += err
        swarm[id].results[i] = float(lines[i])
        print(str(lines[i]))
        log_file.write(str(lines[i]) + " ")
        log_file.write("\n")
        data_file.write(str(lines[i]) + " ")
    log_file.close()
    data_file.write("\n")
    data_file.close()
    return error


def explode(bbid, config):  # explode the bird population
    print("Exploding")
    for i in range(config["num_birds"]):
        if (i != bbid):
            new_bird = Case(i, config)
            # for i in range(dim):
            swarm[i].var = copy.copy(new_bird.var)
            swarm[i].velocity = copy.copy(new_bird.velocity)
            swarm[i].best_part_var = copy.copy(new_bird.best_part_var)
        else:
            print("Didnt change bird ID " + str(i))


# def crossover():  # working on develpoing cross-over feature
#     nTop = 6
#     top_birds = []
#     bottom_birds = []
#     slice_var_at = dim / 2
#     sortedbirds = sortbirds()
#
#     for i in range(num_birds):
#         if (i < nTop):
#             top_birds.append(sortedbirds[i].id)
#             bottom_birds.append(sortedbirds[num_birds - 1 - i])
#
#     for i in range(nTop / 2):
#         for j in range(dim):
#             if (j < slice_var_at):
#                 swarm[bottom_birds[i]].var[j] = copy.copy(swarm[top_birds[i]].var[j])
#                 swarm[bottom_birds[i + 1]].var[j] = copy.copy(swarm[top_birds[i + 1]].var[j])
#             else:
#                 swarm[bottom_birds[i]].var[j] = copy.copy(swarm[top_birds[i + 1]].var[j])
#                 swarm[bottom_birds[i + 1]].var[j] = copy.copy(swarm[top_birds[i]].var[j])
#
#
# def sortbirds():  # to sort the birds by their error
#     sorted = False
#     swarm1 = swarm[:]
#     while not sorted:
#         sorted = True
#         for i in range(num_birds - 1):
#             if swarm1[i].error > swarm1[i + 1].error:
#                 sorted = False  # We found two birds in the wrong order
#                 temp = swarm1[i + 1]
#                 swarm1[i + 1] = swarm1[i]
#                 swarm1[i] = temp
#     return swarm1


# ------------------------------------

# ------------------------------------

def runann(config):  # run ANN
    command = ['./ANN.sh']  # call the shell file
    p = subprocess.Popen(command, bufsize=2048, shell=True, stdin=subprocess.PIPE)
    p.wait()

    # Read the ANN result file
    ann_result_file = open("param.txt", "r")
    lines = ann_result_file.read().split(' ')
    ann_result_file.close()

    # Write ANN result file in the directory of the assigned bird
    os.chdir(str(config["ann_bird_id"]))
    file = 'new_par.txt'
    text_file = open(file, "w")
    text_file.write("%f %f" % (float(lines[0]), float(lines[1])))  # change this if more/less parameters
    text_file.close()
    os.chdir("..")
    swarm[config["ann_bird_id"]].var[0] = float(lines[0])
    swarm[config["ann_bird_id"]].var[1] = float(lines[1])


def optimize1(var, total, max_range, min_range):
    max_config = max_range
    min_config = min_range
    sum_var: Union[float, int] = sum(var)
    if sum_var <= total:
        value_add = total - sum_var
        for i in range(len(var)):
            var[i] = var[i] + value_add / len(var)
            max_value = max_config[i]
            if var[i] > max_value:
                var[i] = max_value
            # print("Plus", i, var[i])

    else:
        value_add = abs(total - sum_var)
        for i in range(len(var)):
            var[i] = var[i] - value_add / len(var)
            max_value = max_config[i]
            min_value = min_config[i]
            if var[i] > max_value:
                var[i] = max_value
            elif var[i] < min_value:
                var[i] = min_value
            # print("Minus", i, var[i])

        pass

    return var


class Case:
    def __init__(self, id, config):
        self.id = id
        self.rnd = random.Random(id)
        self.var = [0.0 for i in range(config["dim"])]
        self.max = [0.0 for i in range(config["dim"])]
        self.min = [0.0 for i in range(config["dim"])]
        self.velocity = [0.0 for i in range(config["dim"])]
        self.best_part_var = [0.0 for i in range(config["dim"])]
        self.best_part_res = [0.0 for i in range(config["nresults"])]
        self.results = [0.0 for i in range(config["nresults"])]

        for i in range(config["dim"]):
            min = float(copy.copy(config["min_var"][i]))
            max = float(copy.copy(config["max_var"][i]))
            self.var[i] = ((max - min) *
                           self.rnd.random() + min)
            self.velocity[i] = ((max - min) *
                                self.rnd.random() + min)
            self.max[i] = max
            self.min[i] = min

        print("A=", self.var[:])

        for ii in range(5000):
            # print(ii)
            self.var[:] = optimize1(self.var[:], float(copy.copy(config["total"])), self.max[:], self.min[:])
            if (float(copy.copy(config["total"])) - sum(self.var[:]) < 0.0001) and \
                    (float(copy.copy(config["total"])) - sum(self.var[:]) >= 0.0):
                break
        print("B=", self.var[:])

        self.error = 1000.0 + self.rnd.random()
        self.best_part_var = copy.copy(self.var)
        self.best_part_err = self.error  # best error


def Solve(config):  # called once with max iter = max_epochs
    rnd = random.Random(0)
    global epoch
    n = config["num_birds"]
    stuck_counter = 0
    best_bird_id = 0
    prev_best_bird_err = 0.0

    max_list = [0.0 for i in range(config["dim"])]
    min_list = [0.0 for i in range(config["dim"])]
    for i in range(config["dim"]):
        max_list[i] = float(copy.copy(config["max_var"][i]))
        min_list[i] = float(copy.copy(config["min_var"][i]))

    # error=[] # saves errors
    # create n random particles
    for i in range(n):
        swarm.append(Case(i, config))  # fills the swarm with n birds at random positions

    best_swarm_var = [0.0 for i in range(config["dim"])]
    best_swarm_err = sys.float_info.max  # swarm best too far
    for i in range(n):  # check each bird
        if swarm[i].error < best_swarm_err:
            best_swarm_err = swarm[i].error
            best_swarm_var = copy.copy(swarm[i].var)

    epoch = 0
    log_file = open(LOGFILE, "w")
    data_file = open(DATAFILE, "w")
    data_file.close()
    best_file = open(BESTFILE, "w")
    best_file.close()

    while epoch < config["max_epochs"]:  # for each epoch
        out = "Epoch = " + str(
            epoch) + " best id = %d best error = %.3f var1=%.3f var2=%.3f var3=%.3f var4=%.3f var5=%.3f error=%.3f \n" % (
                  best_bird_id, best_swarm_err, swarm[best_bird_id].var[0], swarm[best_bird_id].var[1],
                  swarm[best_bird_id].var[2], swarm[best_bird_id].var[3], swarm[best_bird_id].var[4],
                  swarm[best_bird_id].error)

        verbosity = 1  # how often print the best bird variables and error
        if epoch % verbosity == 0 and epoch > 1:
            show_vector(best_swarm_var)
            print(out)
        log_file = open(LOGFILE, "a")
        log_file.write(out)
        log_file.close()
        best_data = ""
        best_file = open(BESTFILE, "a")
        for j in range(config["dim"]):
            best_file.write("%f " % (swarm[best_bird_id].best_part_var[j]))
        for k in range(config["nresults"]):
            best_file.write("%f " % (swarm[best_bird_id].best_part_res[k]))
        best_file.write("\n")
        best_file.close()
        # Run the set of 25 simulations here
        #   crossover()
        var = runsimulation(config)

        for i in range(n):  # process each bird
            swarm[i].error = readresults(i, config, var)  # read namd outputs and calculate error
            # compute new velocity of curr bird

            total = float(copy.copy(config["total"]))

            for k in range(config["dim"]):
                r1 = rnd.random()  # randomizations
                r2 = rnd.random()

                swarm[i].velocity[k] = ((config["w"] * swarm[i].velocity[k]) +
                                        (config["c1"] * r1 * (swarm[i].best_part_var[k] - swarm[i].var[k])) +
                                        (config["c2"] * r2 * (best_swarm_var[k] - swarm[i].var[k])))

                if swarm[i].velocity[k] > config["max_vel"][k]:
                    swarm[i].velocity[k] = config["max_vel"][k]
                elif swarm[i].velocity[k] < - config["max_vel"][k]:
                    swarm[i].velocity[k] = - config["max_vel"][k]

                # print("velocity ["+str(i)+","+str(k)+"] = "+ str(swarm[i].velocity[k]))

                # compute new variables using new velocity
                # print("variables old [" + str(i) + "," + str(k) + "] = " + str(swarm[i].var[k]))
                swarm[i].var[k] += swarm[i].velocity[k]

                if swarm[i].var[k] < config["min_var"][k]:
                    swarm[i].var[k] = config["min_var"][k]
                elif swarm[i].var[k] > config["max_var"][k]:
                    swarm[i].var[k] = config["max_var"][k]

                # print("variables [" + str(i) + "," + str(k) + "] = " + str(swarm[i].var[k]))
            # compute error of new set of variables
            # swarm[i].error = #runsimulation(swarm[i].var,i,n)
            print("C", swarm[i].var[:])
            for j in range(5000):
                swarm[i].var[:] = optimize1(swarm[i].var[:], float(copy.copy(config["total"])), max_list, min_list)
                if (float(copy.copy(config["total"])) - sum(swarm[i].var[:]) < 0.00001) and \
                        (float(copy.copy(config["total"])) - sum(swarm[i].var[:]) >= 0):
                    break

            print("D", swarm[i].var[:])

            # is new variable a new best for the bird?
            if swarm[i].error < swarm[i].best_part_err:
                swarm[i].best_part_err = swarm[i].error
                swarm[i].best_part_var = copy.copy(swarm[i].var)
                for l in range(config["nresults"]):
                    swarm[i].best_part_res[l] = copy.copy(swarm[i].results[l])

            # is new variable a new best overall?
            if swarm[i].error < best_swarm_err:
                best_swarm_err = swarm[i].error
                best_swarm_var = copy.copy(swarm[i].var)
                best_bird_id = i

        # for-each particle

        epoch += 1
        if prev_best_bird_err == best_swarm_err:
            stuck_counter += 1
        else:
            stuck_counter = 0
        if stuck_counter > config["max_epoch_stuck"]:
            stuck_counter = 0
            explode(best_bird_id, config)

        prev_best_bird_err = best_swarm_err

    # while
    return swarm[best_bird_id]


def launch_pso(subfolder, config):
    subprocess.Popen(
        'seq 0 ' + config['num_birds'] + ' | parallel --jobs 8 --workdir $PWD '
                                         '--sshloginfile nodelist.txt "cd {}/' + subfolder + ' ;'
                                                                                             ' ~/softwares/lammps-17Jan18/src/lmp_serial -in ./in.eam"; wait'
    )


def print_start_message(config):
    print("\nBegin particle swarm optimization\n")

    print("Setting num_particles = " + str(config["num_birds"]))
    print("Setting max_epochs    = " + str(config["max_epochs"]))
    print("\nStarting PSO algorithm\n")


def print_end_message(best_bird):
    print("\nPSO completed\n")
    print("\nBest solution found:")
    show_vector(best_bird.var)
    print("Error of best solution = %.6f" % best_bird.best_part_err)

    print("\nEnd PSO\n")


if __name__ == '__main__':
    config = get_config()
    print_start_message(config)
    best_bird = Solve(config)
    print_end_message(best_bird)
