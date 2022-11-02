import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


c = 299792458

lama = [394.4, 422.2, 440.9, 461.2, 491.5, 516.9, 545.1, 587.8, 624.6]
Spannung = [1.261, 1.041, 0.873, 0.802, 0.635, 0.519, 0.452, 0.224, 0.105]

freq = []

for val in lama:
    freq.append(c / (val*10**(-9)))

print(linregress(freq, Spannung))
plt.plot(freq, Spannung, '.r')
plt.xlabel('Frequency')
plt.ylabel('Voltage')
plt.show()




