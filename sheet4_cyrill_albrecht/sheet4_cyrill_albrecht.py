"""Skeleton for sheet4.

This contains a fit, you don't need to fit, just use the covariance matrix and best fit from the sheet.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# define function for the fit
def linear(x, m, q):
    """Linear function

    Returns the linear function of x with slope m and intercept q.

    .. math::
        f(x) = m x + q

    Args:
        x (float): x value
        m (float): slope
        q (float): y-intercept
    """
    return m * x + q


def fit_linear(x, y):
    """Fit linear function

    Fits the linear function to the data (x, y).

    Args:
        x (array): x values
        y (array): y values
    """
    popt, pcov = curve_fit(linear, x, y)
    return popt, pcov


def ex4():
    # loading data
    data = np.loadtxt('sand.txt')
    # you DON'T need to fit! Just use the covariance matrix and best fit from the sheet.
    # But this is how the fit would work
    diameter = data[:, 0]
    slope = data[:, 1]
    slope_err = data[:, 2]
    m = 16.1
    q = -2.61
    plt.figure()
    plt.errorbar(diameter, slope, yerr=slope_err, fmt='b.')
    plt.plot(diameter, diameter * m + q, 'r-')
    plt.xlabel('diameter sand grain in mm')
    plt.ylabel('slope')
    plt.savefig('ex4_4.pdf')
    # plt.show()

    cov_mat = [[1.068, 0], [0, 0.118]]
    diameter = 1.5
    slope = diameter * m + q
    matrices = np.array([[1.5, 1]])
    matrices_transp = matrices.transpose()
    variance = float(np.dot(np.dot(matrices, cov_mat) , matrices_transp))
    err = abs(np.sqrt(variance))
    print(f"the slope of a beach whose sand grains have the diameter of 1.5 mm is {slope:.2f} +/- {err:.2f}.\n")

    cov_mat = [[1.068, -0.302], [-0.302, 0.118]]
    diameter = 1.5
    slope = diameter * m + q
    matrices = np.array([[1.5, 1]])
    matrices_transp = matrices.transpose()
    variance = float(np.dot(np.dot(matrices, cov_mat), matrices_transp))
    err = abs(np.sqrt(variance))
    print(f"the slope of a beach whose sand grains have the diameter of 1.5 mm is {slope:.2f} +/- {err:.2f}.\n")
    print(f"with the correlation of m and q the uncertainty is smaller. The covariance is negative,as a result the "
          f"uncertainty getting smaller")


if __name__ == '__main__':
    ex4()
