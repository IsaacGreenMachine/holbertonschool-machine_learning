#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

fruits = ["apples", "bananas", "oranges", "peaches"]
colors = ["red", "yellow", "orange", "#ffe5b4"]
names = ["Farrah", "Fred", "Felicia"]
bottom = np.zeros(fruit[0].shape)
for num in range(len(fruit)):
    plt.bar(range(len(fruit[0])), fruit[num], color=colors[num], bottom=bottom, width=0.5)
    bottom = bottom + fruit[num]

plt.ylabel("Quantity of Fruit")
plt.ylim([0, 80])
plt.yticks(range(0, 81, 10))
plt.xticks([0, 1, 2], labels=names)
plt.title("Number of Fruit per Person")
plt.legend(fruits)
plt.savefig('bars')
plt.show()

'''
Alternatively:
plt.bar(range(len(fruit[0])), fruit[0], color="red")
plt.bar(range(len(fruit[0])), fruit[1], color="yellow", bottom=fruit[0])
plt.bar(range(len(fruit[0])), fruit[2], color="orange", bottom=fruit[0]+fruit[1])
plt.bar(range(len(fruit[0])), fruit[3], color="#ffe5b4", bottom=fruit[0]+fruit[1]+fruit[2])
'''
