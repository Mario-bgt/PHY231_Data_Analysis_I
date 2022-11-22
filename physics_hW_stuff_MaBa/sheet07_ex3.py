import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


phi = [15, 30, 45, 60, 75, 105, 120, 135, 150]
dN = [132000, 7800, 1435, 477, 211, 70, 52, 43, 33]
c = 299792458


plt.plot(phi, dN, '.r')
plt.semilogy()
plt.xlabel('Winkel')
plt.ylabel('dN')
plt.show()




