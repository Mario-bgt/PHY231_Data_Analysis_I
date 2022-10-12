import numpy as np
import matplotlib.pyplot as plt

c = 299792458
g = 9.81

t = np.linspace(0, 1e8, 2000)


def v(t):
    return (g*t)/(1+((g**2)*(t**2))/(c**2))


fig = plt.figure()
plt.plot(t, v(t))
plt.show()



