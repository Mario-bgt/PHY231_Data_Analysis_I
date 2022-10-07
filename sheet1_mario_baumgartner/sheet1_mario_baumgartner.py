import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt('ironman.txt')
'''I know its not part of the exersice but I dont want any 0 in my data for obvious reason.'''
file = np.delete(file, np.where((file == 0))[0], axis=0)

total_rank = file[:, 0]
year_of_birth = 2010 - file[:, 1]
total_time = file[:, 2]
swimming_time = file[:, 3]
swimming_rank = file[:, 4]
cycling_time = file[:, 5]
cycling_rank = file[:, 6]
running_time = file[:, 7]
running_rank = file[:, 8]
fig = plt.figure()
plt.plot(total_rank, total_time, '.')
plt.xlabel('total rank')
plt.ylabel('total time in minutes')
plt.savefig('total_rank_vs_total_time.png')
# plt.show()
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
