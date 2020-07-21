#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: style.py
@time: 7/20/20 9:12 PM
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

def plot_use_sns(style=None, i=None):
    """
    :param stlye: dict, None, or one of {default, darkgrid, whitegrid, dark, white, ticks}
    :return:
    """
    if style == "default":
        sns.set()  # Default Style
        plt.subplot(3, 3, i + 1)
    else:
        with sns.axes_style(style):
            plt.subplot(3, 3, i + 1)

        # if style == "ticks":
        #     sns.set_style(style, {"xtick.major.size": 8, "ytick.major.size": 8})
        #     sns.despine() # Remove the top and right spines from plot(s)
        #
        # else:
        #     sns.set_style(style)

    plot_sin(title=style)

def main():
    styles = ["default", "darkgrid", "whitegrid", "dark", "white", "ticks"]
    for i, style in enumerate(styles):
        plot_use_sns(style,i)

    plt.show()

if __name__ == "__main__":

    main()