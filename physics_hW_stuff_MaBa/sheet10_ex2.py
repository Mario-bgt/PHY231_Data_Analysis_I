import numpy as np
import matplotlib.pyplot as plt


def schroed(a, x):
    upper = 2
    lower = np.pi*a**2
    power = (-x ** 2) / (a ** 2)
    return (upper/lower)**(1/4)*np.exp(power)


def potential(a, x):
    upper = 2
    lower = np.pi*a**2
    power = (-x ** 2) / (a ** 2)
    return ((upper/lower)**(1/4)*np.exp(power))**2




def fourier(a, k):
    upper = 2*np.pi
    lower = a ** 2
    power = (-1*(k*a)**2) / 4
    return (upper / lower) ** (1 / 4) * np.exp(power)


values = np.linspace(-2, 2, 1000)

x_values = [v for v in values]
y_values = [schroed(0.25, v) for v in values]
plt.plot(x_values, y_values, '-r', label='a=0.25')
y_values = [potential(0.25, v) for v in values]
plt.plot(x_values, y_values, '-.r', label='a=0.25')
y_values = [schroed(0.5, v) for v in values]
plt.plot(x_values, y_values, '-b', label='a=0.5')
y_values = [potential(0.5, v) for v in values]
plt.plot(x_values, y_values, '-.b', label='a=0.5')
y_values = [schroed(0.75, v) for v in values]
plt.plot(x_values, y_values, '-c', label='a=0.75')
y_values = [potential(0.75, v) for v in values]
plt.plot(x_values, y_values, '-.c', label='a=0.75')
y_values = [schroed(1, v) for v in values]
plt.plot(x_values, y_values, '-k', label='a=1')
y_values = [potential(1, v) for v in values]
plt.plot(x_values, y_values, '-.k', label='a=1')
plt.grid()
plt.legend()
plt.savefig('sheet10_ex2.jpeg')
plt.show()

