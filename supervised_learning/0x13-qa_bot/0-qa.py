#!/usr/bin/env python3
"""  0. Question Answering """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    tz = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tz = tz.tokenize(question)
    reference_tz = tz.tokenize(reference)
    tokens = ['[CLS]'] + question_tz + ['[SEP]'] + reference_tz + ['[SEP]']

    input_ids = tz.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    type_ids = [0] * (len(question_tz) + 2) + [1] * (len(reference_tz) + 1)
    input_ids, input_mask, type_ids = map(
        lambda x:
            tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.int32), 0),
            (input_ids, input_mask, type_ids)
    )
    outputs = model([input_ids, input_mask, type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    return tz.convert_tokens_to_string(answer_tokens)
