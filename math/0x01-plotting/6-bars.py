#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


width = 0.5
colors = ['r', 'yellow', '#ff8000', '#ffe5b4']
names = ['Farrah', 'Fred', 'Felicia']
bottom = np.zeros(3)


for elem, color in zip(fruit, colors):
    if color == 'r':
        plt.bar(names, elem, width, bottom=bottom, color=color, label='apples')
        bottom += elem
    elif color == "yellow":
        plt.bar(names, elem, width, bottom=bottom,
                color=color, label='bananas')
        bottom += elem
    elif color == "#ff8000":
        plt.bar(names, elem, width, bottom=bottom,
                color=color, label='oranges')
        bottom += elem
    else:
        plt.bar(names, elem, width, bottom=bottom,
                color=color, label='peaches')
        bottom += elem

plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.title('Number of Fruit per Person')

plt.show()
