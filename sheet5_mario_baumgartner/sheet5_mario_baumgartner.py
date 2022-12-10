"""Datenanalysis sheet 4 of Mario Baumgartner"""
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
    chi = 0
    for x, y, err in zip(x, y, err):
        chi += ((x - y) ** 2) / err ** 2
    return chi


def ex_1a():
    """Run exercise 1a."""
    data = np.loadtxt("current_measurements.txt")
    v = data[:, 0]
    a = data[:, 1]
    plt.figure()
    plt.errorbar(v, a, yerr=0.2, fmt='ok')
    plt.grid()
    plt.title('Current plotted against the voltage')
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')
    plt.savefig('ex1a.png')
    # plt.show()
    plt.close()
    print("ex1a executed.")


def chi2_1b(R):
    """Calculate chi2 in dependence of the resistance."""
    data = np.loadtxt("current_measurements.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    err = [0.2 for i in voltage]
    current_pred = current_ohmslaw(voltage, R)
    chi2val = chi2(current, current_pred, err)
    return chi2val


def ex_1c():
    """Run exercise 1c."""
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi2_1b(i) for i in resistances]
    best_R = resistances[chi2val.index(min(chi2val))]
    print(f"The best Chi 2 value is {min(chi2val):.3f} corresponding to {best_R:.3f} Ohm")
    print("ex1b executed.")
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.grid()
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("Chi 2")
    plt.title("Chi 2 as a function of resistance")
    plt.savefig("ex1c.png")
    # plt.show()
    plt.close()
    print("ex1c executed.")


def ex_1d():
    data = np.loadtxt("current_measurements.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    mean_voltage = np.mean(voltage)
    mean_current = np.mean(current)
    mean_voltage_times_current = np.mean(voltage * current)
    R = (np.mean(voltage ** 2) - mean_voltage ** 2) / (mean_voltage_times_current - (mean_current * mean_voltage))
    print(f"The obtained value with the analytical approach is {R:.2f} Ohm, which does not agree with the 1.75 Ohm"
          f" from 1c, thus unfortunately the fit is not very good.")
    print("ex1d executed.")


def ex_1e():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi2_1b(i) for i in resistances]
    best_R = resistances[chi2val.index(min(chi2val))]
    err_chi = min(chi2val) + 1
    approved_chi = [i for i in chi2val if i > err_chi]
    err_R = abs(resistances[chi2val.index(min(approved_chi))] - best_R)
    print(f"The error upon chi 2 is {err_chi:.2f} which means our error upon R is +/- {err_R:.4f} and thus still "
          f"doesnt explain the difference.")
    print("ex1e executed.")


def chi2_2a(R):
    """Calculate chi2 in dependence of the resistance with varying uncertainty."""
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    err = data[:, 2]
    current_pred = current_ohmslaw(voltage, R)
    chi2val = chi2(current, current_pred, err)
    return chi2val


def ex_2b():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi2_2a(i) for i in resistances]
    best_R = resistances[chi2val.index(min(chi2val))]
    print(f"The best Chi 2 value is {min(chi2val):.3f} corresponding to {best_R:.3f} Ohm")
    print("ex2a executed.")
    plt.figure()
    plt.plot(resistances, chi2val)
    plt.grid()
    plt.xlabel("Resistance [Ohm]")
    plt.ylabel("Chi 2")
    plt.title("Chi 2 as a function of resistance")
    plt.savefig("ex2b.png")
    # plt.show()
    plt.close()
    print(f"The minimum value of chi 2 is {min(chi2val):.2f} and the best fit value for R is {best_R:.2f} Ohm. ")
    print("ex2b executed.")


def ex_2c():
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    err = data[:, 2]
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi2_2a(i) for i in resistances]
    best_R = resistances[chi2val.index(min(chi2val))]
    slope = 1 / best_R
    plt.figure()
    plt.errorbar(voltage, current, yerr=err, fmt='ok', label='measurements')
    x_values = np.linspace(min(voltage), max(voltage), 500)
    y_values = [i * slope for i in x_values]
    plt.plot(x_values, y_values, '-.r', label='best line fit')
    plt.legend()
    plt.grid()
    plt.title('Best line fit plottet with the data')
    plt.xlabel('voltage [V]')
    plt.ylabel('current [A]')
    plt.savefig('ex2c.png')
    # plt.show()
    plt.close()
    print(f"It does look like a quite good fit for large values, the lower ones are higher than the fit.")
    print("ex2c executed.")


def ex_2d():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi2_2a(i) for i in resistances]
    delta_chi = min(chi2val) + 1
    best_R = resistances[chi2val.index(min(chi2val))]
    approved_chi = [i for i in chi2val if i > delta_chi]
    err_R = abs(resistances[chi2val.index(min(approved_chi))] - best_R)
    print(f"The obtained error with the chi square rule is {best_R:.2f} +/- {err_R:.2f} Ohm which is not "
          f"compatible with 2 Ohm.")
    print("ex2d executed.")


def chi2_bias(x, y, err, e_bias):
    chi_bias = 0
    for x, y, err in zip(x, y, err):
        chi_bias += ((x - y - e_bias) ** 2) / err ** 2
    return chi_bias


def chi_2e(R):
    e_bias = 0.7
    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    err = data[:, 2]
    current_pred = current_ohmslaw(voltage, R)
    chi2val = chi2_bias(current, current_pred, err, e_bias)
    return chi2val


def ex_2e():
    resistances = np.linspace(1.6, 2, 100)
    chi2val = [chi_2e(i) for i in resistances]
    delta_chi = min(chi2val) + 1
    best_R = resistances[chi2val.index(min(chi2val))]
    approved_chi = [i for i in chi2val if i > delta_chi]
    err_R = abs(resistances[chi2val.index(min(approved_chi))] - best_R)
    print(f"The obtained error with the chi square rule is {best_R:.2f} +/- {err_R:.2f} Ohm which is accurate and "
          f"compatible with the expected 2 Ohm.")
    print("ex2e executed.")


def ex_2f():
    resistances = np.linspace(1.6, 2, 100)
    chi_unc = [chi2_2a(i) for i in resistances]
    chi_bias = [chi_2e(i) for i in resistances]
    min_chi_unc = min(chi_unc)
    min_chi_bias = min(chi_bias)
    # ndf = number of data - number of parameters
    ndf1 = 5
    ndf2 = 4
    goodness_of_fit1 = min_chi_unc / ndf1
    goodness_of_fit2 = min_chi_bias / ndf2
    p1 = scipy.stats.chi2.pdf(min_chi_unc, ndf1)
    p2 = scipy.stats.chi2.pdf(min_chi_bias, ndf2)
    print(f"The goodness of the fit without bias is {goodness_of_fit1:.3f}.\n"
          f"The goodness of the fit with bias is {goodness_of_fit2:.3f}.\n"
          f"The chance to get a better fit without the bias is {p1 * 100:.1f}%, which is way to low to be acceptable.\n"
          f"The chance to get a better fit with the bias is {p2 * 100:.1f}%, which is a very good result\n")
    print("ex2f executed.")


def ex_2g_2h():
    """Run exercise 2g."""

    def function(x, e, r):
        return e + x / r

    data = np.loadtxt("current_measurements_uncertainties.txt")
    voltage = data[:, 0]
    current = data[:, 1]
    popt, pcov = opt.curve_fit(function, voltage, current)
    print(f"The optimised parameters are: e_bias = {popt[0]:.3f} and R = {popt[1]:.3f}")
    print("ex2g executed.")
    err_r = np.sqrt(pcov[1][1])
    err_e = np.sqrt(pcov[0][0])
    corr_coeff = pcov[0][1] / (err_e * err_r)
    print(f"The curve fit found that the error upon R is {err_r:.3f} and the error upon e_bias = {err_e:.3f}.\n"
          f"Their correlation coefficient is equal to {corr_coeff:.3f}.\n"
          f"The obtained error in 2e) is quite close to the one obtained here.")
    print("ex2h executed.")


if __name__ == '__main__':
    ex_1a()
    ex_1c()
    ex_1d()
    ex_1e()
    ex_2b()
    ex_2c()
    ex_2d()
    ex_2e()
    ex_2f()
    ex_2g_2h()
