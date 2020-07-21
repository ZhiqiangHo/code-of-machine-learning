#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: regression.py
@time: 7/21/20 9:09 AM
@desc:
'''
import matplotlib.pyplot as plt
import seaborn as sns


def show_data(data, style):
    """
    # show the data of type dataframe
    :param data: data of type dataframe
    :return:
    """
    for i, var in enumerate(data):
        plt.subplot(3,3,i+1)
        if data[var].dtype == 'float64':
            sns.regplot(x=var, y='tip', data=data)
        elif data[var].dtype == 'int64':
            sns.regplot(x=var, y="tip", data=data, x_jitter=0.05)
        elif data[var].dtype == 'category':
            if style == "strip":
                sns.stripplot(x=var, y="total_bill", data=data, jitter=True)
            elif style == "swarm":
                sns.swarmplot(x=var, y="total_bill", hue="time", data=data)
            elif style == "box":
                sns.boxplot(x=var, y="tip", hue='time', data=data)
            elif style == "violin":
                sns.violinplot(x=var, y="total_bill", hue="sex", data=data, split=True)
            elif style == "bar":
                sns.barplot(x=var, y="total_bill", hue="sex",data=data)
            elif style == "point":
                sns.pointplot(x=var, y="total_bill", hue="sex", data=data, markers=["o", "x"],linestyles=["-", "--"])
            # elif style == "factor":
            #     sns.factorplot(x='sex', y="total_bill", data=data, kind='bar')
    plt.show()

if __name__ == '__main__':
    sns.set(style="whitegrid", color_codes=True)
    tips_data = sns.load_dataset('tips')
    styles = ["strip", "swarm", "box", "violin", "bar", "point"]
    show_data(tips_data, style=styles[5])
