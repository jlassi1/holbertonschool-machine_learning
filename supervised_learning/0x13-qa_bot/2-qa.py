#!/usr/bin/env python3
"""  2. Answer Questions  """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def answer_loop(reference):
    """function that answers questions from a reference text"""
    tz = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q:', end='')
        question = input()
        if question.lower() in exit_list:
            print('A: Goodbye')
            break
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
        if not answer_tokens:
            print('A: Sorry, I do not understand your question.')
        else:
            print('A:' + tz.convert_tokens_to_string(answer_tokens))
