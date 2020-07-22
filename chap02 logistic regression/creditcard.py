#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: creaditcard.py
@time: 7/22/20 8:57 AM
@desc:
'''

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("white")

class Plot(object):

    def plot_class_distribution(self, dataframe, class_name):
        count_class = pd.value_counts(values=dataframe[class_name]) # return Series
        sns.barplot(x=count_class.index, y=count_class.values)
        plt.title("Creditcard class histogram")
        plt.show()

    def plot_confusion_matrix(self, model, test_data, test_label, title='Confusion Matrix'):
        y_pre = model.predict(X=test_data)
        cnf_matrix = confusion_matrix(test_label, y_pre)
        sns.heatmap(cnf_matrix, annot=True, xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

class data_process(Plot):
    def __init__(self, method=None):
        self.method = method

    def get_data(self, dataframe):
        dataframe["NormAmount"] = StandardScaler().fit_transform(
            dataframe["Amount"].values.reshape(-1, 1))     # Normal Features
        data = dataframe.drop(["Time", "Amount"], axis=1)  # del un-useful information

        if self.method == "un_sample":
            x_train, x_test, y_train, y_test = self.un_sample(dataframe=data, plot_un_sample=False)
            return x_train, x_test, y_train, y_test

        elif self.method == "original":
            x_train, x_test, y_train, y_test = self.orin_data(dataframe=data)
            return x_train, x_test, y_train, y_test

        elif self.method == "SMOTE":
            x_train, x_test, y_train, y_test = self.smote_data(dataframe=data)
            return x_train, x_test, y_train, y_test

    def un_sample(self, dataframe, plot_un_sample=False):
        """
        # under sample
        :param dataframe:
        :param plot_un_sample: if plot the num of positive and negative
        :return:
        """
        pos_num = dataframe[dataframe.Class == 1].shape[0]
        pos_data = dataframe[dataframe["Class"] == 1]
        neg_data = dataframe[dataframe["Class"] == 0].sample(n=pos_num, replace=True)
        data = pd.merge(pos_data, neg_data, how="outer")
        if plot_un_sample == True:
            super().plot_class_distribution(dataframe=data, class_name="Class")
        un_sample_data = shuffle(data)

        x, y = self.__split_x_y(un_sample_data)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def smote_data(self, dataframe):
        from imblearn.over_sampling import SMOTE
        x, y = self.__split_x_y(dataframe)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        oversample = SMOTE(random_state=0)
        X_train, Y_train = oversample.fit_sample(x_train, y_train)
        return X_train, x_test, Y_train.reshape(-1,1), y_test

    def orin_data(self, dataframe):
        data = shuffle(dataframe)
        x, y = self.__split_x_y(data)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def __split_x_y(self,dataframe):
        x = dataframe.ix[:, dataframe.columns != "Class"].as_matrix()
        y = dataframe.ix[:, dataframe.columns == "Class"].as_matrix()
        return x, y

def main(show_class_dis=False, is_plot_confusion_matrix=True, method="SMOTE"):
    plot = Plot()
    df = pd.read_csv("creditcard.csv")
    if show_class_dis==True:
        plot.plot_class_distribution(dataframe=df, class_name="Class")

    x_train, x_test, y_train, y_test = data_process(method=method).get_data(df)

    fold = KFold(n_splits=5, shuffle=False)
    for iter, (train_ind, test_ind) in enumerate(fold.split(x_train)):
        lr = LogisticRegression(C=1.0, penalty="l2", max_iter=1000)
        lr.fit(X=x_train[train_ind,:],y=y_train[train_ind,:].ravel())
        pre = lr.predict(X=x_train[test_ind,:])
        # pre_prob = lr.predict_proba(X=x_train[test_ind,:])

        recall_acc = recall_score(y_train[test_ind], pre) # Recall = TP/(TP+FN)
        print("iter {}, recall acc {}".format(iter, recall_acc))

        if is_plot_confusion_matrix:
            plot.plot_confusion_matrix(model=lr, test_data=x_test, test_label=y_test)

if __name__ == '__main__':

    main()