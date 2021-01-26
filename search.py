from pulp import *
import numpy as np
import struct
from array import array
from os.path import join
import random
import pandas as pd
import os
import sys
from classification_functions import MnistDataloader

class Pair:
    def __init__(self, x, y) -> None:
        self.index = x
        self.distance = y


def f(pair):
        return pair.distance


def manhattan_distance(p, q):
    return sum([abs(int(p[i])-int(q[i])) for i in range(len(p))])


def divideToClusters(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def EMD(sig_train, sig_test):
    c_train = []
    w_train = []

    for j in range(16):
        c_train.append(sig_train[j][0])
        w_train.append(sig_train[j][1])

    c_test = []
    w_test = []

    for j in range(16):
        c_test.append(sig_test[j][0])
        w_test.append(sig_test[j][1])

    c_train = np.array(c_train)
    w_train = np.array(w_train)
    c_test = np.array(c_test)
    w_test = np.array(w_test)
    
    distances = np.zeros((len(c_train), len(c_test)))
    for i in range(len(c_train)):
        for j in range(len(c_test)):
            distances[i][j] = np.linalg.norm(int(c_train[i]) - int(c_test[j]))

    variablesList = []
    for i in range(len(c_train)):
        tempList = []
        for j in range(len(c_test)):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = LpProblem("EMD", LpMinimize)

    constraint = []
    objectiveFunction = []
    for i in  range(len(c_train)):
        for j in range(len(c_test)):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])
            constraint.append(variablesList[i][j])

    problem += lpSum(objectiveFunction)
    tempMin = min(np.sum(w_train), np.sum(w_test))              # constraints
    problem += lpSum(constraint) == tempMin

    for i in range(len(c_train)):
        constraint1 = [variablesList[i][j] for j in range(len(c_test))]
        problem += lpSum(constraint1) <= w_train[i]

    for j in range(len(c_test)):
        constraint2 = [variablesList[i][j] for i in range(len(c_train))]
        problem += lpSum(constraint2) <= w_test[j]

    problem.solve(GLPK_CMD(msg = False))

    flow = value(problem.objective.valueOrDefault())
    return flow / tempMin


if __name__ == '__main__':
    if (len(sys.argv) == 12):
        i = 0
        for var in sys.argv:                                        # get values from command line
            if (var == "-d"):
                input_original = sys.argv[i + 1]
            if (var == "-q"):
                queries_original = sys.argv[i + 1]
            if (var == "-l1"):
                labels_input = sys.argv[i + 1]
            if (var == "-l2"):
                labels_queries = sys.argv[i + 1]
            if (var == "-o"):
                output_file = sys.argv[i + 1]

            i = i + 1
    else:
        print("Wrong input. Using default values.")

        input_original = 'train-images-idx3-ubyte'          # default values if not given by user
        queries_original = 't10k-images-idx3-ubyte'
        labels_input = 'train-labels-idx1-ubyte'
        labels_queries = 't10k-labels-idx1-ubyte'
        output_file = 'results_emd.txt'

    (xtrain, ytrain) = MnistDataloader(input_original, labels_input)    # read datasets
    (xtest, ytest) = MnistDataloader(queries_original, labels_queries)

    x_train = np.array(xtrain)
    x_test = np.array(xtest)
    y_train = np.array(ytrain)
    y_test = np.array(ytest)

    x_train2d = np.reshape(x_train, (len(x_train), 28, 28))
    x_test2d = np.reshape(x_test, (len(x_test), 28, 28))

    signatures_train = [[[0 for k in range(2)] for j in range(16)] for i in range(len(x_train2d))]
    sum_train = []
    i = 0
    for image in x_train2d:
        temp = divideToClusters(image, 7, 7)
        
        for j in range(16):
            signatures_train[i][j][1] = np.sum(temp[j])             # w
            signatures_train[i][j][0] = temp[j][3][3]               # p

        i += 1

    signatures_test = [[[0 for k in range(2)] for j in range(16)] for i in range(len(x_test2d))]
    i = 0
    for image in x_test2d:
        temp = divideToClusters(image, 7, 7)
        
        for j in range(16):
            signatures_test[i][j][1] = np.sum(temp[j])              # w
            signatures_test[i][j][0] = temp[j][3][3]                # p

        i += 1

    pVec_emd = []
    pVec_manhattan = []
    average_correct_emd = []
    average_correct_manhattan = []

    # for q in range(len(x_test)):
    #     for i in range(len(x_train)):
    for q in range(10):
        for i in range(1000):
            dist_emd = EMD(signatures_train[i], signatures_test[q])
            pVec_emd.append(Pair(i, dist_emd))

            dist_manhattan = manhattan_distance(x_test[q], x_train[i])
            pVec_manhattan.append(Pair(i, dist_manhattan))

        pVec_emd.sort(key = f)
        pVec_manhattan.sort(key = f)

        correct_emd = 0
        correct_manhattan = 0
        for j in range(10):
            if (y_test[q] == y_train[pVec_emd[j].index]):
                correct_emd += 1
            if (y_test[q] == y_train[pVec_manhattan[j].index]):
                correct_manhattan += 1

        average_correct_emd.append(float(correct_emd)*10)
        average_correct_manhattan.append(float(correct_manhattan)*10)

    average_emd = sum(average_correct_emd)/len(average_correct_emd)
    average_manhattan = sum(average_correct_manhattan)/len(average_correct_manhattan)

    f = open("results_emd.txt", "w")
    f.write("Average Correct Search Results EMD: " + str(average_emd) + "\n")
    f.write("Average Correct Search Results MANHATTAN: " + str(average_manhattan) + "\n")

    f.close()