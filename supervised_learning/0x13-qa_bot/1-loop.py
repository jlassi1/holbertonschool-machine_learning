#!/usr/bin/env python3
""" 1. Create the loop """


def loop_QA():
    """function that takes in input from the user
    with the prompt Q: and prints A: as a response"""
    exit_list = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q:', end='')
        question = input()
        if question.lower() in exit_list:
            print('A: Goodbye')
            break
        print('A:',)


if __name__ == "__main__":
    loop_QA()
