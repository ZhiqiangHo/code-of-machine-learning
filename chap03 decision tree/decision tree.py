#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: decision tree.py
@time: 7/23/20 8:38 AM
@desc:
'''

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import tree
import pandas as pd


sns.set_style("white")

def get_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    df_y = pd.DataFrame(data=housing.target, columns=housing.target_names)
    df[str(df_y.columns.values[0])] = df_y

    x = df[["Latitude", "Longitude"]].as_matrix()
    y = df[["MedHouseVal"]].as_matrix()
    x_train, x_text, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=0)
    return x_train, x_text, y_train, y_test

def write_tree_model(model, filename):
    # sudo apt-get install graphviz
    import pydotplus
    dot_data = tree.export_graphviz(decision_tree=model, out_file=None,
                                feature_names=["Latitude", "Longitude"], filled=True, impurity=False, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(filename)

def main(is_write_grid=False):
    x_train, x_text, y_train, y_test = get_data()

    # sklearn.grid_search can find parameter
    model = tree.DecisionTreeRegressor(max_depth=2)

    model.fit(X=x_train, y=y_train)
    score = model.score(x_text, y_test)
    print("score is {}".format(score))

    if is_write_grid:
        write_tree_model(model, "tree_model.png")


if __name__ == '__main__':
    main()

