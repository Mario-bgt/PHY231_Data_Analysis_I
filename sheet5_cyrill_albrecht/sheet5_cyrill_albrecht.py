"""Skeleton for Data Analysis Fall 2022 sheet 5.

This sheet helps to structure the code for the exercises of sheet 5.
It shows idioms and best practices for writing code in Python.

You can directly use this sheet and modify it.

(It is not guaranteed to be bug free)
"""
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt


def current_ohmslaw(U, R):
    r"""Calculate the current according to Ohm's Law given the voltage U and resistance R.

    Ohm's Law states that the current is given by:

    .. math::

        I = \frac{U}{R}

    Args:
        U (float, array): The measured voltage.
        R (float, array): The resistance.

    Returns:
        float or array: Value of the linear function. Shape is the broadcast shape of
            the inputs.
    """
    return U / R


def current_ohmslaw_bias(U, R, bias=None):
    """Calculate the current according to Ohm's Law given the voltage U and resistance R with a bias.

    Ohm's Law states that the current is given by:

    .. math::

        I = \frac{U}{R}

    We can add a bias to the current by adding a constant to the voltage:

    .. math::

        I = \frac{U + bias}{R}

    Args:
        U (float, array): The measured voltage.
        R (float, array): The resistance.
        bias (float, array): The bias to add to the voltage. If None, no bias is added.

    Returns:
        float or array: Value of the linear function. Shape is the broadcast shape of
            the inputs.
    """
    if bias is None:
        bias = 0  # with this, we can also use the function without bias.
    return current_ohmslaw(U + bias, R)


def chi2(x, y, err):
    """Calculate the chi2 statistic for a dataset and its predictions.

    Args:
        x (array): The first data set.
        y (array): Predicted values for the first data set.
        err (array): The error on the measurements of the first data set.

    Returns:
        float: The chi2 statistic.
    """
    chi2 = 0
    for i, j in enumerate(x):
        chi2 = chi2 + (((j - y[i]) ** 2) / (err[i] ** 2))
    return chi2



def ex_1a():
    """Run exercise 1a."""

    # Here is your code for exercise 1a.
    data = np.loadtxt('current_measurements.txt')
    V = data[:,0]
    A = data[:,1]
    plt.figure()
    plt.errorbar(V, A, yerr=0.2, fmt='s')
    # plt.plot(V, A)
    plt.grid()
    plt.title('Current as a function of the voltage')
    plt.xlabel('Voltage values [V]')
    plt.ylabel('Current value [A]')
    plt.savefig('ex1_a')
    plt.show()
    plt.close()

    print("ex1a executed.")


# This is an example of creating 1b composing different functions together.
def chi2_1b(R):
    """Calculate chi2 in dependence of the resistance."""

    # Here is your code for exercise 1b.
    data = np.loadtxt("current_measurements.txt")  # load it from file for example
    voltage = data[:, 0]
    current = data[:, 1]
    error = [0.2 for i in voltage]# etc.
    current_pred = current_ohmslaw(voltage, R)
    chi2val = chi2(current, current_pred, error)
    return chi2val


def ex_1c():
    """Run exercise 1c."""
    # Here is your code for exercise 1c.
    resistances = np.linspace(1.6, 2, 100)  # start, stop, number of steps
    chi2val = 0
    for i in resistances:
        chi2val += chi2_1b(i)
    best_resistance = resistances[chi2val.index(min(chi2val))]
    print(f"The best Chi 2 value is {min(chi2val):.4f} for {best_resistance:.4f} Ohm")
    print("ex1b executed.")
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.grid()
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("Chi 2")
    plt.title("Chi 2 as a function of resistance")
    plt.savefig("ex1_c.png")
    # plt.show()
    plt.close()
    print("ex1c executed.")


def ex_2g():
    """Run exercise 2g."""
    # Here we need to use scipy.optimize.curve_fit to fit the data.
    # make sure to first read the documentation for curve_fit
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    # NOTE: curve_fit already calculates the chi2 value (including error) for us!
    # hint: maybe look for simple examples around or play around if it is not clear on how to use curve_fit.

    popt, pcov = opt.curve_fit(...)
    print("ex2g executed.")


if __name__ == '__main__':
    # You can uncomment the exercises that you don't want to run. Here we have just one,
    # but in general you can have more.
    ex_1a()
    ex_1c()
    ex_2g()
    plt.show()  # comment out when submitting
