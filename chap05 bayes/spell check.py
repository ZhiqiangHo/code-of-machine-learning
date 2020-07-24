#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: spell check.py
@time: 7/24/20 9:14 AM
@desc:
'''
"""
argmax_{r} P(r|e) = argmax_{r} P(e|r)p(r)
r: the right words
e: the error words
"""
import re
import collections
letter= 'abcdefghijklmnopqrstuvwxyz'


def edit_dis0(words, all_word):
    return set(w for w in words if w in all_word)

def edit_dis1(word):
    length = len(word)
    return set([word[0:i] + word[i+1:] for i in range(length)] +                           # deletion
               [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(length-1)] +  # transposition
               [word[0:i] + c + word[i+1:] for i in range(length) for c in letter] + # alteration
               [word[0:i] + c + word[i:] for i in range(length+1) for c in letter])  # insertion
def edit_dis2(word, all_word):
    return set(e2 for e1 in edit_dis1(word) for e2 in edit_dis1(e1) if e2 in all_word)


def data_process(filename):
    f = open(filename).read()

    # Remove special characters & Convert to lowercase
    clean_data = re.findall('[a-z]+', f.lower())

    # compute prior probability p(r)
    words_count = collections.defaultdict(lambda :1)
    for word in clean_data:
        words_count[word] += 1

    return clean_data, words_count


def main(word):
    clean_data, words_count = data_process("big.txt")
    # TODO there may be some bug
    candidates_word = edit_dis0([word], clean_data) or edit_dis0(edit_dis1(word), clean_data) or edit_dis2(word, clean_data) or [word]
    return max(candidates_word, key=lambda word: words_count[word])

if __name__ == '__main__':

    print(main("morw"))