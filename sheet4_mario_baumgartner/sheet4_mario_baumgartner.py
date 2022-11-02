"""Datenanalysis sheet 4 of Mario Baumgartner"""

import numpy as np
import matplotlib.pyplot as plt


def ex4():
    # Part a)
    data = np.loadtxt('sand.txt')
    diameter = data[:, 0]
    slope = data[:, 1]
    slope_err = data[:, 2]
    m = 16.1
    q = -2.61
    plt.figure()
    plt.errorbar(diameter, slope, yerr=slope_err, fmt='b.')  # plot the data with errorbars
    plt.plot(diameter, diameter * m + q, 'r-')  # plot the fit
    plt.xlabel('diameter of the sand grain [mm]')
    plt.ylabel('slope of the beach')
    plt.title("slope of the beach plotted against the sand grain diameter")
    plt.savefig('ex04_4.pdf')
    # plt.show()

    # Part b)
    cov_mat = [[1.068, 0],
               [0, 0.118]]
    dia = 1.5
    y = dia*m + q
    mat = np.array([[1.5, 1]])
    mat_T = np.array([[1.5], [1]])
    var = float((mat@cov_mat@mat_T))
    err = abs(var**.5)
    print(f"The slope for a beach with grain the size of 1.5mm in diameter is {y:.2f} +/- {err:.2f} when neglecting "
          f"the correlation between m and q.\n")

    # part c)
    cov_mat = [[1.068, -0.302],
               [-0.302, 0.118]]
    dia = 1.5
    y = dia*m + q
    mat = np.array([[1.5, 1]])
    mat_T = np.array([[1.5], [1]])
    var = float((mat@cov_mat@mat_T))
    err = abs(var**.5)
    print(f"The slope for a beach with grain the size of 1.5mm in diameter is {y:.2f} +/- {err:.2f} when taking the "
          f"correlation between m and q into account.\n")
    print(f"\nA smaller uncertainty was obtained when taking the correlation between m and q into account.\n This is "
          f"not inuitive since taking the covariance into account one would expect the uncertainty to increase.\n "
          f"However since the covariance is negative the uncertainty decreases")


if __name__ == '__main__':
    ex4()
