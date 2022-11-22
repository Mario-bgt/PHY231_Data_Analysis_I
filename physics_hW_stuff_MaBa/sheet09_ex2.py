import numpy as np
import matplotlib.pyplot as plt


def intesitiy(v, s, lama):
    upper = (np.sin(np.pi*(s/lama)*np.sin(v))**2)
    lower = (np.pi*(s/lama)*np.sin(v))**2
    return upper/lower


values = np.linspace(-1, 1, 1000)
s = 3.2*10**(-6)
lama = 644*10**(-9)
x_values = [(s/lama)*np.sin(v) for v in values]
y_values = [intesitiy(np.sin(v), s, lama) for v in values]
plt.plot(x_values, y_values, '-g')
plt.grid()
plt.savefig('sheet09_ex2.jpeg')
plt.show()
