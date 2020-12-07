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

# your code here

fig = plt.figure()
fig.tight_layout(pad=10)

fig.suptitle('All in one')


ax0 = plt.subplot2grid((3, 2), (0, 0))
#ax0 = fig.add_subplot(321, xmargin=5)
ax0.plot(y0, 'r')

ax1 = plt.subplot2grid((3, 2), (0, 1))
#ax1 = fig.add_subplot(322)
ax1.scatter(x1, y1, color='#f504c9', marker='.')
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title('Men\'s Height vs Weight', fontsize='x-small')


ax2 = plt.subplot2grid((3, 2), (1, 0))
#ax2 = fig.add_subplot(323, xmargin=5)
ax2.plot(x2, y2)
plt.xlim(0, 28650)
plt.yscale('log')
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of C-14', fontsize='x-small')

ax3 = plt.subplot2grid((3, 2), (1, 1))
#ax3 = fig.add_subplot(324, xmargin=5)
ax3.plot(x3, y31, 'r--', label='C-14')
ax3.plot(x3, y32, 'g', label='Ra-226')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title('Exponential Decay of Radioactive Elements', fontsize='x-small')
plt.legend(fontsize='x-small')

ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
#ax4 = fig.add_subplot(313, xmargin=5)
plt.xticks(range(0, 101, 10))
ax4.hist(student_grades, bins=[i for i in range(
    0, 101, 10)], range=10, edgecolor='k')
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')


plt.subplots_adjust(wspace=0.30, hspace=.8)
plt.show()
