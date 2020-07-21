#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: style detail.py
@time: 7/20/20 10:32 PM
@desc:
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_sin(title=None):
    """
    # Create Date
    :return: plot
    """

    x = np.linspace(0,2*math.pi, 100)
    for i in range(7):
        plt.plot(x, np.sin(x+i*0.3))
        plt.title(str(title))

def despine_detail(style="ticks",detail=None):
    if detail == "origin":
        sns.set_style(style)
        plot_sin(title=style)
    elif detail == "offset":
        plot_sin(title=detail)
        sns.despine(offset=10)
    elif detail == "left":
        plot_sin(title=detail)
        sns.despine(left=True)

def context_detail(detail):
    """
    # some detail(figure size, line width etc.) had been encapsulation
    :param detail:
    :return:
    """
    sns.set_context(detail)
    plt.figure(figsize=(8,6))
    plot_sin(title=detail)

def main():
    despine_details = ["origin", "offset", "left"]
    for i, detail in enumerate(despine_details):
        plt.figure()
        # plt.subplot(3,3,i+1)
        despine_detail(detail=detail)
    plt.show()

    context_details = ["paper", "talk", "poster", "notebook"]
    for i, detail in enumerate(context_details):
        context_detail(detail)
    plt.show()

if __name__ == "__main__":
    main()