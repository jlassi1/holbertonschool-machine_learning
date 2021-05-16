#!/usr/bin/env python3
"""  2. Cumulative N-gram BLEU score """
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU
    score for a sentence
    """
    bp = brevity_penalty(sentence, references)
    ngrams = []
    for i in range(1, n + 1):
        ngrams.append(ngram_bleu(references, sentence, i))
    ngrams = np.array(ngrams)
    score = np.exp(np.sum((1 / n) * np.log(ngrams)))
    return bp * score


def ngram_bleu(references, sentence, n):
    """function that calculates the n-gram BLEU score for a sentence"""
    s = ngrams(sentence, n)
    r = list(ngrams(ref, n) for ref in references)
    words = count_clip_ngram(s, r)
    p = sum(words.values()) / len(s)
    return p


def ngrams(sentences, n):
    """ split the sebtence into ngrams"""
    ngrams_sentence = []
    for i in range(len(sentences) - n + 1):
        ngrams_sentence.append(' '.join(sentences[i:i + n]))
    return ngrams_sentence


def brevity_penalty(candidate, references):
    """
    Brevity Penalty
    BP={1 if c>r or exp(1−r/c)if c≤r
    c: length of candidate translation

    r: effective reference length
    """
    c = len(candidate)
    r = np.array([len(r) for r in references])
    r = np.argmin(np.abs(r - c))
    r = len(references[r])

    if c > r:
        return 1
    else:
        return np.exp(1 - float(r) / c)


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
