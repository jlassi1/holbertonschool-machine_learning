#!/usr/bin/env python3
""" 2. Train Word2Vec """
from gensim.models import FastText


def fasttext_model(
        sentences,
        size=100,
        min_count=5,
        window=5,
        negative=5,
        cbow=True,
        iterations=5,
        seed=0,
        workers=1):
    """function that creates and trains a gensim fasttext_model"""
    model = FastText(sentences,
                     size=size,
                     min_count=min_count,
                     negative=negative,
                     window=window,
                     sg=not cbow,
                     iter=iterations,
                     seed=seed,
                     workers=workers)
    # model.train(
    #     sentences,
    #     total_examples=model.corpus_count,
    #     epochs=model.epochs)

    return model
