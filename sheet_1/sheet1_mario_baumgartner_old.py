import numpy as np
import matplotlib.pyplot as plt

total_rank = []
year_of_birth = []
total_time = []
swimming_time = []
swimming_rank = []
cycling_time = []
cycling_rank = []
running_time = []
running_rank = []

file = np.loadtxt('ironman.txt')

for row in file:
    total_rank.append(float(row[0]))
    year_of_birth.append(2010 - float(row[1]))
    total_time.append(float(row[2]))
    swimming_time.append(float(row[3]))
    swimming_rank.append(float(row[4]))
    cycling_time.append(float(row[5]))
    cycling_rank.append(float(row[6]))
    running_time.append(float(row[7]))
    running_rank.append(float(row[8]))

fig = plt.figure()
plt.plot(total_rank, total_time, 'r--')
plt.xlabel('total rank')
plt.ylabel('total time in minutes')
plt.savefig('total_rank_vs_total_time.png')
plt.show()
plt.cla()
plt.plot(year_of_birth, total_time, '.')
plt.xlabel('age')
plt.ylabel('total time in minutes')
plt.savefig('age_vs_total_time.png')
# plt.show()
plt.cla()
plt.plot(running_time, swimming_time, '.')
plt.xlabel('running time in minutes')
plt.ylabel('swimming time in minutes')
plt.savefig('running_vs_swimming.png')
# plt.show()
plt.cla()
plt.plot(swimming_time, total_time, '.')
plt.xlabel('swimming time in minutes')
plt.ylabel('total time in minutes')
plt.savefig('swimming_vs_total.png')
# plt.show()
plt.cla()
plt.plot(cycling_time, total_time, '.')
plt.xlabel('cycling time in minutes')
plt.ylabel('total time in minutes')
plt.savefig('cycling_vs_total.png')
# plt.show()
plt.cla()
plt.plot(running_time, total_time, '.')
plt.xlabel('running time in minutes')
plt.ylabel('total time in minutes')
plt.plot('running_vs_total.png')
# plt.show()
plt.cla()
plt.hist(total_time, range=[450, 1000], bins=np.linspace(450, 1000, 23))
plt.xlabel('total time in minutes')
plt.ylabel('amount of people')
plt.savefig('total_time_hist.png')
# plt.show()
plt.cla()
plt.hist(year_of_birth, range=[15, 17], bins=np.linspace(15, 75, 13))
plt.xlabel('age')
plt.ylabel('amount of people')
plt.savefig('age_hist.png')
# plt.show()
plt.cla()
