#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: singal variable.py
@time: 7/21/20 8:10 AM
@desc:
'''
import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt

def single_var(x):
    """"
    # the knowledge parameter of distplot()
    """
    sns.set(color_codes=True)
    sns.distplot(x,bins=20, kde=False) # Flexibly plot a univariate distribution of observations.
    plt.show()

def multi_var(data):
    df = pd.DataFrame(data, columns=['x1', 'x2'])
    with sns.axes_style("white"):
        sns.jointplot(x="x1",y="x2", data=df, kind="scatter", color="b")
    plt.show()

def iris_pair():
    iris = sns.load_dataset("iris")
    sns.pairplot(iris)
    plt.show()
def main():

    np.random.seed(0)
    x = np.random.normal(size=100)
    single_var(x)

    # Draw random samples from a multivariate normal distribution.
    mean, cov = [0, 1], [(1, 0.5), (0.5, 1)]
    data = np.random.multivariate_normal(mean, cov, 100)  # shape =(100, 2)
    multi_var(data)

    iris_pair()
if __name__ == '__main__':
    main()
