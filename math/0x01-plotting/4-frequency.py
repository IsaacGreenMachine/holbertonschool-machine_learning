#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

for i in range(0, 110, 10):
    plt.bar(
        i,
        (
            (student_grades > i) & (student_grades < i + 9.99)).sum(),
        color="cyan",
        edgecolor="black",
        width=10,
        align="edge"
    )
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.xlim([0, 100])
plt.ylim([0, 30])
plt.xticks(range(0, 110, 10))
plt.show()
