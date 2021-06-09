#!/usr/bin/env python3
"""  3. Semantic Search  """
import tensorflow_hub as hub
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """function  that performs semantic search on a corpus of documents """
    url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(url)
    sentence_search = [sentence]
    documents = []
    for file in os.listdir(corpus_path):
        if file.endswith(".md"):
            filename = os.path.join(corpus_path, file)
            with open(filename) as f:
                doc = f.read()
                documents.append(doc)
    sentence_search_encoded = embed(sentence_search).numpy()
    embed_encoded = embed(documents).numpy()
    corr = np.dot(sentence_search_encoded, embed_encoded.T)

    argmax = np.argmax(corr)
    return documents[argmax]
