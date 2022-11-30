"""Datenanalysis sheet 4 of Mario Baumgartner"""
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
    return U/R


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
    return current_ohmslaw(U+bias, R)


def chi2(x, y, err):
    """Calculate the chi2 statistic for a dataset and its predictions.

    Args:
        x (array): The first data set.
        y (array): Predicted values for the first data set.
        err (array): The error on the measurements of the first data set.

    Returns:
        float: The chi2 statistic.
    """

    return


def ex_1a():
    """Run exercise 1a."""
    data = np.loadtxt("current_measurements.txt")
    v = data[:, 0]
    a = data[:, 1]
    plt.figure()
    plt.scatter(v, current_ohmslaw(v, a))
    plt.xlabel('Voltage')
    plt.ylabel('Current')
    plt.savefig('current_measurements_as_a_function_of_the_voltage.png')
    plt.show()
    # Here is your code for exercise 1a.
    print("ex1a executed.")


# This is an example of creating 1b composing different functions together.
def chi2_1b(R):
    """Calculate chi2 in dependence of the resistance."""

    # Here is your code for exercise 1b.
    data = np.loadtxt("current_measurements.txt")
    voltage = data[:, 0]  # etc.
    current = data[:, 1]
    err = np.array([0.2 for i in voltage])
    current_pred = current_ohmslaw(voltage, current)
    chi2val = chi2(voltage, current, err)
    return chi2val


def ex_1c():
    """Run exercise 1c."""

    # Here is your code for exercise 1c.
    resistances = np.linspace(...)  # start, stop, number of steps
    chi2val = chi2_1b(...)
    # maybe use a for-loop to calculate chi2 for different resistances.


    # plot the chi2 value as a function of the resistance.
    plt.figure()  # ALWAYS create a new figure before plotting.
    plt.plot(...)
    ...  # don't forget to add the legend, labels, titles etc.
    plt.savefig("ex1c.png")
    plt.close()  # ALWAYS close the figure when not needed anymore


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
