#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, axs = plt.subplots(3, 2)
# top left
axs[0][0].plot(y0, color="red")
axs[0][0].set_xlim([0, 10])

# top right
axs[0][1].scatter(x1, y1, c="Magenta")
axs[0][1].set_xlabel("Height (in)")
axs[0][1].set_ylabel("Weight (lbs)")
axs[0][1].set_title("Men's Height vs Weight")

# middle left
axs[1][0].plot(x2, y2)
axs[1][0].set_xlabel("Time (years)")
axs[1][0].set_ylabel("Fraction Remaining")
axs[1][0].set_title("Exponential Decay of C-14")
axs[1][0].set_xlim([0, 28650])
axs[1][0].set_yscale("log")

# middle right
axs[1][1].set_xlabel("Time (years)")
axs[1][1].set_ylabel("Fraction Remaining")
axs[1][1].set_title("Exponential Decay of Radioactive Elements")
axs[1][1].set_xlim([0, 20000])
axs[1][1].set_ylim([0, 1])
axs[1][1].plot(x3, y31, '--r', x3, y32, '-g')
axs[1][1].legend(['C-14', 'Ra-226'])

# bottom:
axs[2][0] = plt.subplot2grid((3, 2), (2, 0), colspan=2)
for i in range(0, 110, 10):
    plt.bar(i, ((student_grades > i) & (student_grades < i + 9.99)).sum(),
            color="cyan", edgecolor="black", width=10, align="edge")
axs[2][0].set_xlabel("Grades")
axs[2][0].set_ylabel("Number of Students")
axs[2][0].set_title("Project A")
axs[2][0].set_xlim([0, 100])
axs[2][0].set_ylim([0, 30])
axs[2][0].set_xticks(range(0, 110, 10))

plt.savefig('all_in_one.png')
plt.show()
'''
plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
'''
