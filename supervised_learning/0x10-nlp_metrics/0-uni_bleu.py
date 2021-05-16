#!/usr/bin/env python3
"""NLP METRICS"""
import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence"""
    unigrams = len(sentence)
    token = np.array([len(r) for r in references])
    idx = np.argmin(np.abs(token - unigrams))
    r = len(references[idx])
    bp = 1
    if r > unigrams:
        bp = np.exp(1 - r / unigrams)
    words = {}
    for i in sentence:
        for ref in references:
            if i in words:
                if words[i] < ref.count(i):
                    words.update({i: ref.count(i)})
            else:
                words.update({i: ref.count(i)})
    p = sum(words.values()) / unigrams
    return bp * p
