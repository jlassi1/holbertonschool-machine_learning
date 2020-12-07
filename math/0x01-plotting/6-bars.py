#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

width = 0.5
colors = ['r', 'yellow', '#ff8000', '#ffe5b4']
labels= ['apples', 'bananas', 'oranges', 'peaches']
names = ['Farrah', 'Fred', 'Felicia']
bottom = np.zeros(3)

for elem, color, label in zip(fruit, colors, labels):
    plt.bar(names, elem, width, bottom=bottom, color=color, label=label)
    bottom += elem

plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.legend()
plt.title('Number of Fruit per Person')

plt.show()
