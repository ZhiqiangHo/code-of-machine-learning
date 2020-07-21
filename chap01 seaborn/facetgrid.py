#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: facetgrid.py
@time: 7/21/20 12:24 PM
@desc:
'''
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")
print(tips.head())

# g = sns.FacetGrid(tips, col='time')
# g.map(plt.hist, "tip")
# plt.show()

g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=0.7)


plt.show()
