#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: ensemble learning.py
@time: 7/23/20 4:42 PM
@desc:
'''

import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd



def data_process(df):
    """
    # split data to : x_train, x_test, y_train, y_test
    :param df: dataframe
    :return:
    """
    # fill none with median value
    df["age"] = df["age"].fillna(df["age"].median())

    # replace the str with int
    sex_class = df["sex"].unique()
    df = df.replace({"sex":sex_class[0]}, 0).replace({"sex":sex_class[1]},1)

    # fill none with "S"
    df["embarked"] = df["embarked"].fillna("S")
    embarked_class = df["embarked"].unique()
    df = df.replace({"embarked":embarked_class[0]}, 0).replace({"embarked":embarked_class[1]}, 1).replace({"embarked":embarked_class[2]},2)

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    x = df[features].as_matrix()
    y = df["survived"].as_matrix().reshape(-1, 1)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return train_test_split(x, y, test_size=0.2, random_state=0)

def main(method="RandomForest"):
    # load data
    dataframe = sns.load_dataset("titanic")
    x_train, x_test, y_train, y_test = data_process(df=dataframe)

    if method == "RandomForest":
        model = RandomForestClassifier(n_estimators=20, min_samples_split=4, min_samples_leaf=2)
    elif method == "Logist":
        model = LogisticRegression(max_iter=1000)
    elif method == "LinearRegression":
        model = LinearRegression()
    else:
        raise Exception("Please Input Method")

    model.fit(x_train, y_train.ravel())

    if False:
        score = model.score(x_test,y_test)
    else:
        score = cross_validate(estimator=model, X=x_test, y=y_test.ravel())
    print("score is {}".format(score))



if __name__ == '__main__':
    main(method="RandomForest")