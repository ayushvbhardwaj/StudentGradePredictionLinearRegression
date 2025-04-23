import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("score.csv")
training_range = 0.8

data_train = data[:int(training_range*len(data))]
data_test = data[int(training_range*len(data)):]

x_train, y_train = data_train["Hours"], data_train["Scores"]
x_test, y_test = data_test["Hours"], data_test["Scores"]


def cost_fn(X, y, w, b):
    total_cost = 0
    for i in range(len(X)):
        cost = ((w*X[i] + b) - y[i])**2
        total_cost += cost
    return (1/(2*len(X)))*total_cost


def partial_derivatives(X, y, w, b):
    dj_dw = 0
    dj_db = 0

    for i in range(len(X)):
        cost = ((w*X[i] + b) - y[i])*X[i]
        dj_dw += cost
    dj_dw = (1/(len(X)))*dj_dw

    for i in range(len(X)):
        cost = ((w*X[i] + b) - y[i])
        dj_db += cost
    dj_db = (1/(len(X)))*dj_db

    return dj_dw, dj_db


def compute_partial_derivatives(X, y, w, b):
    h = 1e-10

    dj_dw = (cost_fn(X, y, w+h, b) - cost_fn(X, y, w, b))/h
    dj_db = (cost_fn(X, y, w, b+h) - cost_fn(X, y, w, b))/h

    return dj_dw, dj_db


def gradient_descent(X, y, w, b, itern, alpha):

    for i in range(itern):
        dj_dw, dj_db = partial_derivatives(X, y, w, b)
        dj_dw1, dj_db1 = compute_partial_derivatives(X, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost = cost_fn(X, y, w, b)

    return w, b, cost


def init():
    X = x_train
    y = y_train
    w = 0
    b = 0
    itern = 10000
    alpha = 0.01

    w, b, cost = gradient_descent(X, y, w, b, itern, alpha)

    print(w, b, cost)

    print('predict')

    error = 0

    y_predicted = []
    for x in (x_test):
        # print(x_test[i])
        y_predict = x*w + b
        y_predicted.append(y_predict)

        error += abs(y_test - y_predict)

        print(y_predict)

    print('g', 8*w + b)

    print('real')
    print(y_test)

    print(f"Error:{error}")

    plt.scatter(X, y, c='r')
    plt.scatter(x_test, y_test, c='pink')
    plt.plot(x_test, y_predicted)
    plt.show()


init()
