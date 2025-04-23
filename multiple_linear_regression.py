import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("student-mat.csv", sep=';')

features = ["sex", "age", "address", "famsize", "traveltime", "studytime",
            "failures", "activities", "romantic", "internet", "famrel", "goout",
            "Dalc", "Walc", "health", "absences", "G1", "G2"]

label = "G3"

data["sex"] = data["sex"].map({"F": 1, "M": 2})
data["address"] = data["address"].map({"U": 1, "R": 2})
data["famsize"] = data["famsize"].map({"GT3": 1, "LE3": 2})
data["activities"] = data["activities"].map({"no": 0, "yes": 1})
data["internet"] = data["internet"].map({"no": 0, "yes": 1})
data["romantic"] = data["romantic"].map({"no": 0, "yes": 1})


training_range = 0.75
data_train = data[:int(training_range*len(data))]
data_test = data[int(training_range*len(data)):]

x_train, y_train = data_train[features], data_train[label]
x_test, y_test = data_test[features], data_test[label]

x_train = x_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(x_test.head())
print(y_test.head())

# print(np.array(x_train.head().iloc[1]))


def cost_func(X, y, w, b):
    cost = 0
    for i in range(len(X)):
        #print(np.dot(np.array(X.iloc[i]), w))
        f_wb_i = np.dot(np.array(X.iloc[i]), w) + b
        cost += (f_wb_i - y[i])**2
    cost = cost/(2*len(X))
    return cost


#print(cost_func(x_train, y_train, np.ones(len(features)), 1))

def compute_derivatives(X, y, w, b):
    dj_dw = np.zeros(len(features))
    dj_db = 0
    for i in range(len(X)):
        # print(np.array(X.iloc[[i]]))
        # print(w)
        f = (np.dot(X.iloc[i], w) + b - y[i])
        f1 = (np.dot(X.T[i], w) + b - y[i])
        #print(f, f1)

        dj_dw_i = 0
        for j in range(len(features)):
            dj_dw_i += f * X[features[j]][i]

        dj_dw[j] = dj_dw_i

        dj_db += f

    dj_dw = dj_dw/(len(X))
    dj_db = dj_db/(len(X))

    return dj_dw, dj_db


# def compute_derivatives(X, y, w, b):
#     h = 1e-10

#     dj_dw = (cost_func(X, y, w+h, b) - cost_func(X, y, w, b))/h
#     dj_db = (cost_func(X, y, w, b+h) - cost_func(X, y, w, b))/h

#     return dj_dw, dj_db


# def compute_derivatives(X, y, w, b):
#     m = len(X)  # Number of training examples

#     # Compute predictions
#     predictions = np.dot(X, w) + b
#     errors = predictions - y

#     # Compute derivatives
#     dj_dw = (1/m) * np.dot(X.T, errors)  # Vectorized gradient for weights
#     dj_db = (1/m) * np.sum(errors)       # Scalar gradient for bias

#     return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, itern):
    for i in range(itern):
        dj_dw, dj_db = compute_derivatives(X, y, w, b)
        # print(w)
        # print(dj_dw)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cost = cost_func(X, y, w, b)
        cost1 = cost_func(x_test, y_test, w, b)
        #print(i, cost)

        if i % 1000 == 0:
            print(i, cost, cost1)

    return w, b, cost


def init():
    X = x_train
    y = y_train
    w = np.random.randn(len(features)) * 0.01
    b = 0
    print(w, b)
    alpha = 0.0001
    itern = 10000

    w, b, cost = gradient_descent(X, y, w, b, alpha, itern)
    cost1 = cost_func(x_test, y_test, w, b)

    print(cost, cost1)

    print(w, b)


init()
