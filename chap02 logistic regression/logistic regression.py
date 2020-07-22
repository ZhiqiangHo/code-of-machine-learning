#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: logistic regression.py
@time: 7/21/20 3:30 PM
@desc:
'''
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def show_data(frame_data):
    positive = frame_data[frame_data["species"] == 1]
    negative = frame_data[frame_data["species"] == 0]

    sns.swarmplot(x="sepal_length", y="petal_length", hue="species", data=positive, palette="Set2")
    sns.swarmplot(x="sepal_length", y="petal_length", hue="species", data=negative, palette="Set1")
    plt.show()

def data_process(data_name=None, is_showdata=None):
    df = sns.load_dataset(data_name)
    iris_data = df[["sepal_length", "petal_length", "species"]].replace({"species": "setosa"}, 0).replace(
        {"species": "virginica"}, 1)

    iris_data = iris_data[~iris_data["species"].isin(["versicolor"])] # del the row of species = versicolor
    iris_data.insert(0, "ones", 1)
    if is_showdata == True:
        show_data(iris_data)
    all_data = np.array(iris_data.as_matrix(), dtype=np.float64)
    np.random.shuffle(all_data)
    x = all_data[:,:-1].reshape(-1,3)
    y = all_data[:, -1].reshape(-1,1)

    return x, y

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def model(x, theta):
    """
    :param x: feature 1, x1, x2
    :param theta: theta0-theta2
    :return: theta1 + theta1*x1 + theta2*x2
    """

    return sigmoid(np.dot(x, theta.T))

def gradient(x, y, theta):
    error = model(x,theta)-y
    gradient = np.sum(np.multiply(error, x), axis=0) / len(x)
    return gradient


def cost(x, y, theta):
    pos = np.multiply(-y, np.log(model(x,theta)))
    neg = np.multiply((1-y), np.log(1-model(x, theta)))
    return np.sum(pos-neg) / len(x)

def main(data_name, is_showdata=True):

    x, y = data_process(data_name, is_showdata)
    thetas = np.ones([1, 3])
    lr = 0.01
    costs = []
    for i in range(200):
        loss = cost(x,y, thetas)
        grad = gradient(x,y,thetas)
        thetas = thetas - lr*grad
        costs.append(loss)
    sns.set_style("white")
    plt.plot(costs)
    plt.title("loss")
    plt.show()

if __name__ == '__main__':

    main(data_name='iris', is_showdata=False)

