#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: new class.py
@time: 7/24/20 11:16 AM
@desc:
'''
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import jieba
import jieba.analyse
import csv

def data_process(new_file="val.txt", stopwords_file="stopwords.txt"):

    df_news = pd.read_table(new_file, names=["category", "theme", "URL", "content"], encoding="utf-8")

    df_stopwords = pd.read_table(stopwords_file, index_col=False, sep='\t', names=["stopword"], encoding='utf-8', quoting=csv.QUOTE_NONE,)

    # select the value of "content"
    content = df_news["content"].values.tolist()

    # split words
    content_split = [jieba.lcut(i) for i in content if len(jieba.lcut(i)) >1 and jieba.lcut(i) != '\r\n']

    # converted to dataframe
    df_content = pd.DataFrame({"content_split":content_split})
    return df_content, df_stopwords, df_news

def drop_stopwords(df_contents, df_stopwords):

    # convert data & stop_words to list
    contents = df_contents["content_split"].values.tolist()
    stopwords = df_stopwords["stopword"].values.tolist()
    content_clean = []
    all_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_clean.append(str(word))
        content_clean.append(line_clean)
    return content_clean, all_clean

def plot_wordcloud(all_clean):
    from wordcloud import WordCloud
    from matplotlib import pyplot as plt
    import matplotlib
    import numpy as np
    df_all_clean = pd.DataFrame({"all_clean": all_clean})
    words_count = df_all_clean.groupby(by=["all_clean"])["all_clean"].agg({"count": np.size})
    words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
    matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
    wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=80)
    word_frequence = {x[0]: x[1] for x in words_count.head(100).values}
    wordcloud = wordcloud.fit_words(word_frequence)
    plt.imshow(wordcloud)
    plt.show()


def LDA(content_clean):
    import gensim
    # LDA:
    # Input: list of list -> [[], []]
    dictionary = gensim.corpora.Dictionary(content_clean)
    corpus = [dictionary.doc2bow(sentence) for sentence in content_clean]

    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    print(lda.print_topic(1, topn=5))
    for topic in lda.print_topics(num_topics=20, num_words=5):
        print(topic[1])

def vector(data):
    words = []
    for line_index in range(len(data)):
        try:
            words.append(" ".join(data[line_index]))
        except:
            print(line_index)
    return words

def main(is_wordcloud=False, is_LDA=False):
    # get data & stop_words
    df_content, df_stopwords, df_news= data_process()

    # del stop_words
    content_clean, all_clean = drop_stopwords(df_contents=df_content, df_stopwords=df_stopwords)

    if is_LDA:
        LDA(content_clean)

    if is_wordcloud:
        plot_wordcloud(all_clean)

    df_train = pd.DataFrame({"contents_clean":content_clean, "label":df_news["category"]})
    label_mapping = {'汽车': 0, '财经': 1, '科技': 2, '健康': 3, '体育': 4, '教育': 5, '文化': 6, '军事': 7, '娱乐': 8, '时尚': 9}
    df_train["label"] = df_train["label"].map(label_mapping)
    x_train, x_test, y_train, y_test = train_test_split(df_train["contents_clean"].values, df_train["label"].values,
                                                        random_state=1)

    x_train_vec = vector(x_train)
    x_test_vec = vector(x_test)

    vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vec.fit(x_train_vec)
    
    model = MultinomialNB()
    model.fit(vec.transform(x_train_vec), y_train)

    score = model.score(vec.transform(x_test_vec), y_test)

    print(score)

if __name__ == '__main__':
    main()