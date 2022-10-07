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
    if 0 not in row:
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
plt.plot(total_rank, total_time, 'ro')
plt.xlabel('Rang')
plt.ylabel('Zeit in Minuten')
plt.savefig('Rang_vs_Zeit.png')
# plt.show()
plt.cla()

plt.plot(year_of_birth, total_time, 'ro')
plt.xlabel('Alter in Jahren')
plt.ylabel('Zeit in Minuten')
plt.savefig('Alter_vs_Zeit')
# plt.show()
plt.cla()

plt.plot(running_time, swimming_time, 'ro')
plt.xlabel('Laufzeit in Minuten')
plt.ylabel('Schwimmzeit in Minuten')
plt.savefig('Laufzeit_vs_Schwimmzeit')
# plt.show()
plt.cla()

plt.plot(swimming_time, total_time, 'ro')
plt.xlabel('Schwimmzeit in Minuten')
plt.ylabel('Zeit in Minuten')
plt.savefig('Schwimmzeit_vs_Zeit')
# plt.show()
plt.cla()

plt.plot(cycling_time, total_time, 'ro')
plt.xlabel('Fahrzeit in Minuten')
plt.ylabel('Zeit in Minuten')
plt.savefig('Fahrzeit_vs_Zeit')
# plt.show()
plt.cla()

plt.plot(running_time, total_time, 'ro')
plt.xlabel('Laufzeit in Minuten')
plt.ylabel('Zeit in Minuten')
plt.savefig('Laufzeit_vs_Zeit')
# plt.show()
plt.cla()

plt.hist(total_time, range=[450, 1000], bins=25)
plt.xlabel('Zeit in Minuten')
plt.ylabel('Anzahl Personen')
plt.savefig('Zeit_Histogramm.png')
# plt.show()
plt.cla()

plt.hist(year_of_birth, range=[15, 75], bins=13)
plt.xlabel('Alter in Jahren')
plt.ylabel('Anzahl Personen')
plt.savefig('Alter_Histogramm.png')
# plt.show()
plt.cla()
