#!/usr/bin/env python3
""" 0. Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """function  that calculates the unigram BLEU score for a sentence"""
    c = len(sentence)
    r = np.argmin(abs(len(r) - c) for r in references)
    r = len(references[r])
    if c > r:
        bp = 1
    else:
        bp = np.exp(1 - float(r) / c)

    words = count_clip_ngram(sentence, references)
    p = sum(words.values())
    return bp * p / c


def count_clip_ngram(sentence, references):
    """
    Countclip=min(Count,Max_Ref_Count)
    """
    words = dict()
    for word in sentence:
        for ref in references:
            if word in words:
                if words[word] < ref.count(word):
                    words.update({word: ref.count(word)})
            else:
                words.update({word: ref.count(word)})
    return words
