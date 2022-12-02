"""Skeleton for Data Analysis Fall 2022 sheet 5.

This sheet helps to structure the code for the exercises of sheet 5.
It shows idioms and best practices for writing code in Python.

You can directly use this sheet and modify it.

(It is not guaranteed to be bug free)
"""
import numpy as np
import scipy.optimize as opt
import scipy.stats
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
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_1b(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    print(f"The best Chi^2 value is {min(chi2val):.4f} for {best_resistance:.4f} Ohm")
    print("ex1b executed.")
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.grid()
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("Chi^2")
    plt.title("Chi^2 as a function of resistance")
    plt.savefig("ex1_c.png")
    plt.show()
    plt.close()
    print("ex1c executed.")

def ex_1d():
    data = np.loadtxt("current_measurements.txt")
    V = data[:, 0]
    I = data[:, 1]
    mean_V = np.mean(V)
    mean_I = np.mean(I)
    mean_V_x_I = np.mean(V * I)
    R = (np.mean(V ** 2) - (mean_V ** 2)) / (mean_V_x_I - (mean_I * mean_V))
    print(f"The analytical approach is {R:.3f} Ohm and doesnt agree with 1.75 Ohm"
          f" so its not a good fit in 1c.")
    print("ex1d executed.")

def ex_1e():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_1b(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    error_chi = min(chi2val) + 1
    chi_right = []
    for i in chi2val:
        if i > error_chi:
            chi_right.append(i)
    error_resistance = abs(resistances[chi2val.index(min(chi_right))] - best_resistance)
    print(f"The error of chi^2 is {error_chi:.2f} and the error on R is +/- {error_resistance:.4f} which doesnt tell "
          f"us something about the difference")
    print("ex1e executed.")


def chi2_2(R):
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    error = data[:, 2]
    current_ohm = current_ohmslaw(voltage, R)
    chi2val = chi2(current, current_ohm, error)
    return chi2val


def ex_2a():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_2(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    print(f"The best Chi^2 value is {min(chi2val):.4f} for {best_resistance:.4f} Ohm")
    print("ex2a executed.")


def ex_2b():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_2(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.xlabel('Resistances [Ohm]')
    plt.ylabel('Chi^2')
    plt.title('Chi^2 as a function of the resistance')
    plt.grid()
    plt.savefig('ex2_b')
    plt.show()
    plt.close()
    print(f"Minimum value of chi^2 = {min(chi2val):.2f}, best fit value for R = {best_resistance:.2f} Ohm. ")
    print('ex2b executed')


def ex_2c():
    data = np.loadtxt('current_measurements_uncertainties.txt')
    current = data[:, 1]
    voltage = data[:, 0]
    error = data[:, 2]
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_2(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    plt.figure()
    plt.errorbar(voltage, current, yerr=error, fmt='s')
    x_val = np.linspace(min(voltage), max(voltage))
    y_val = []
    for i in x_val:
        y_val.append(i * (1 / best_resistance))
    plt.plot(x_val, y_val, '-g')
    plt.grid()
    plt.title('Best line fit on plot of measurements')
    plt.xlabel('voltage [V]')
    plt.ylabel('current [A]')
    plt.savefig('ex2_c')
    plt.show()
    plt.close()


def ex_2d():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi2_2(i))
    best_resistance = resistances[chi2val.index(min(chi2val))]
    delta_chi = min(chi2val) + 1
    chi_right = []
    for i in chi2val:
        if i > delta_chi:
            chi_right.append(i)
    error_resistance = abs(resistances[chi2val.index(min(chi_right))] - best_resistance)
    print(f"The error with the chi^2 rule is {best_resistance:.2f} +/- {error_resistance:.2f} Ohm and this is not "
          f"compatible.")
    print("ex2d executed.")


def chi2_bias(x, y, error, e_bias):
    chi_bias = 0
    for i, j in enumerate(x):
        chi_bias += ((j - y[i] - e_bias) ** 2) / (error[i] ** 2)
    return chi_bias

def chi_2e(R):
    e_bias = 0.7
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    error = data[:, 2]
    current_ohm = current_ohmslaw(voltage, R)
    chi2val = chi2_bias(current, current_ohm, error, e_bias)
    return chi2val


def ex_2e():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = []
    for i in resistances:
        chi2val.append(chi_2e(i))
    delta_chi = min(chi2val) + 1
    best_resistance = resistances[chi2val.index(min(chi2val))]
    chi_right = []
    for i in chi2val:
        if i > delta_chi:
            chi_right.append(i)
    error_resistance = abs(resistances[chi2val.index(min(chi_right))] - best_resistance)
    print(f"The error with the chi^2 rule is {best_resistance:.2f} +/- {error_resistance:.2f} Ohm and this is "
          f"compatible.")
    print("ex2e executed.")


def ex_2f():
    resistances = np.linspace(1.6, 2, 100)
    chi_uncertainty = []
    for i in resistances:
        chi_uncertainty.append(chi2_2(i))
    chi_bias = []
    for i in resistances:
        chi_bias.append(chi_2e(i))
    chi_uncertainty_min = min(chi_uncertainty)
    chi_bias_min = min(chi_bias)
    ndf1 = 5
    ndf2 = 4
    fit1_goodness = chi_uncertainty_min / ndf1
    fit2_goodness = chi_bias_min / ndf2
    probability1 = (scipy.stats.chi2.pdf(chi_uncertainty_min, ndf1)) * 100
    probability2 = (scipy.stats.chi2.pdf(chi_bias_min, ndf2)) * 100
    print(f"Without bias the goodness of fit is {fit1_goodness:.3f}.")
    print(f"With bias the goodness of fit is {fit2_goodness:.3f}.")
    print(f"to get a better fit without bias the probability is {probability1:.2f}%.")
    print(f"to get a better fit with bias the probability is {probability2:.2f}%.")
    print("ex2f executed.")


def function(x, e, r):
    return e + x / r


def ex_2g_and_2h():
    """Run exercise 2g."""
    # Here we need to use scipy.optimize.curve_fit to fit the data.
    # make sure to first read the documentation for curve_fit
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # NOTE: curve_fit already calculates the chi2 value (including error) for us!
    # hint: maybe look for simple examples around or play around if it is not clear on how to use curve_fit.
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    popt, pcov = opt.curve_fit(function, voltage, current)
    print(f"e_bias = {popt[0]:.3f}, R = {popt[1]:.3f}")
    print("ex2g executed.")
    variance_resistance = pcov[1][1]
    error_resistance = np.sqrt(variance_resistance)
    variance_error = pcov[0][0]
    error_e = np.sqrt(variance_error)
    covariance_re = pcov[0][1]
    correlation_coefficient = covariance_re / (error_e * error_resistance)
    print(f"the error on the resistance is {error_resistance:.3f}")
    print(f"the error on e_bias is {error_e:.3f}")
    print(f"The correlation coefficient is {correlation_coefficient:.3f}.")
    print("ex2h executed.")


if __name__ == '__main__':
    # You can uncomment the exercises that you don't want to run. Here we have just one,
    # but in general you can have more.
    ex_1a()
    ex_1c()
    ex_1d()
    ex_1e()
    ex_2a()
    ex_2b()
    ex_2c()
    ex_2d()
    ex_2e()
    ex_2f()
    ex_2g_and_2h()

