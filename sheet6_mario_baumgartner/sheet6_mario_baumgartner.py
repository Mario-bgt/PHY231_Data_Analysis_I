"""Datenanalysis sheet 4 of Mario Baumgartner"""
import math

import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import math
import scipy.stats as scs


def nll(data, prob, a):
    """Calculate the negative log likelihood for a dataset and its predictions.

    Args:
        data (array): The data set.
        prob (function): probability function for the data set.
        a (float): according alpha value

    Returns:
        float: The negative log likelihood.
    """
    NLL = 0
    for i in data:
        NLL += -math.log(prob(a, i))
    return NLL


def two_nll(data, prob, a):
    """Calculate 2 times the negative log likelihood for a dataset and its probabilities.

    Args:
        data (array): The data set.
        prob (function): probability function for the data set.
        a (float): according alpha value
    Returns:
        float: 2 times the negative log likelihood.
    """
    return 2 * nll(data, prob, a)  # an easy way to re-use existing code.


def binned_2nll(data, prob, a, nbins, integrate=False):
    """Calculate 2 times the negative log likelihood for a dataset and its probabilites.

    Args:
        data (array): The data set.
        prob (function): Probability function.
        a (float): according alpha value
        nbins (int): Number of bins to use.
        integrate (bool): If the value is obtained by summation or integration

    Returns:
        float: 2 times the binned negative log likelihood.
    """
    counts, edges = np.histogram(data, bins=nbins)
    if integrate:
        return None
    else:
        bincenter = (edges[:-1] + edges[1:]) / 2
        nllhist = 0
        for i in range(nbins):
            nllhist += 2 * - (counts[i] * math.log(np.exp(-bincenter[i] / a) * 125 / (a * (1 - np.exp(-5 / a))))) + (
                    np.exp(-bincenter[i] / a) * 125 / (a * (1 - np.exp(-5 / a))))
        return nllhist


def probtemp(tau, t):
    return np.exp(-tau / t) * 125 / (t * (1 - np.exp(-5 / t)))


def ex_1():
    # Part a)
    data = np.loadtxt('MLE.txt')

    def prob_1a(a, x):
        return .5 * (1 + a * x)

    x_val = np.linspace(0, 1, 1000)
    y_val = [nll(data, prob_1a, i) for i in x_val]
    plt.plot(x_val, y_val)
    plt.title('Ex_1a negative log likelihood plot')
    plt.xlabel('alpha')
    plt.ylabel('NLL(alpha)')
    plt.grid()
    plt.show()
    plt.savefig('ex1a.png')
    plt.clf()
    print("ex1a executed.")
    # Part b)
    nll_min = x_val[y_val.index(min(y_val))]
    print(f'The maximum likelihood estimator is {nll_min:.2f}, which one can also see by looking at the plot.')
    print('ex1b executed.')


def ex2():
    # Part a)
    def prob_2(tau, t):
        lower = tau * (1 - np.exp(-5 / tau))
        return (1 / lower) * np.exp(-t / tau)

    data = np.loadtxt('exponential_data.txt')
    x_val = np.linspace(1.8, 2.2, 1000)
    y_val = [two_nll(data, prob_2, i) for i in x_val]
    y_val = [y_val[i] - min(y_val) for i in range(1000)]
    plt.plot(x_val, y_val)
    plt.title('Ex_2a twice negative log likelihood plot with shift to 0')
    plt.xlabel('tau')
    plt.ylabel('2*NLL(tau)')
    plt.grid()
    plt.show()
    plt.savefig('ex2a.png')
    plt.clf()
    print("ex2a executed.")
    # Part b)
    y_binned = [binned_2nll(data, prob_2, i, 40) for i in x_val]
    y_binned = [y_binned[i] - min(y_binned) for i in range(1000)]
    plt.plot(x_val, y_val, color='red', label='Binned 2NLL')
    plt.plot(x_val, y_binned, color='blue', label='2NLL unbinned')
    plt.xlabel('tau [nanoseconds]')
    plt.ylabel('2*NLL(tau)')
    plt.legend()
    plt.title('Exercise 2b) NLL2 shifted and binned')
    plt.grid()
    plt.show()
    plt.savefig('2b_unbinned_binned_2NLL.png')
    plt.clf()
    print('ex2b executed.')
    # Part c)

    plt.plot(x_val, y_val, color='red', label='Binned 2NLL')
    plt.plot(x_val, y_binned, color='blue', label='2NLL unbinned')
    plt.xlabel('tau [nanoseconds]')
    plt.ylabel('2*NLL(tau)')
    plt.legend()
    plt.title('Exercise 2b) NLL2 shifted and binned')
    plt.grid()
    plt.show()
    plt.savefig('2b_unbinned_binned_2NLL.png')
    plt.clf()
    return None


def ex3():
    data = np.loadtxt('polynomial_data.txt')
    counts, edges = np.histogram(data, bins=20)
    bincenter = (edges[:-1] + edges[1:]) / 2
    y_val = counts
    yerr = [np.sqrt(i) for i in counts]
    plt.bar(bincenter, y_val, width=0.08, color='r', yerr=yerr)
    plt.errorbar(bincenter, y_val, yerr=yerr, color="b", drawstyle='steps-mid')
    plt.xlabel('value')
    plt.ylabel('amount of data')
    plt.title('Exercise 3a) binned measurements')
    plt.grid()
    plt.show()
    plt.savefig('3a_histogramm_plot.png')
    plt.clf()
    print('ex3a executed.')

    def fit1(x, a, b):
        return a * x + b

    def fit2(x, a, b, c):
        return a * x ** 2 + b * x + c

    def fit3(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d

    def fit4(x, a, b, c, d, e):
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    popt1, pcov1 = opt.curve_fit(fit1, bincenter, y_val)
    popt2, pcov2 = opt.curve_fit(fit2, bincenter, y_val)
    popt3, pcov3 = opt.curve_fit(fit3, bincenter, y_val)
    popt4, pcov4 = opt.curve_fit(fit4, bincenter, y_val)
    x_values = np.linspace(-1, 1, 1000)
    y_fit1 = [fit1(i, popt1[0], popt1[1]) for i in x_values]
    y_fit2 = [fit2(i, popt2[0], popt2[1], popt2[2]) for i in x_values]
    y_fit3 = [fit3(i, popt3[0], popt3[1], popt3[2], popt3[3]) for i in x_values]
    y_fit4 = [fit4(i, popt4[0], popt4[1], popt4[2], popt4[3], popt4[4]) for i in x_values]
    plt.plot(x_values, y_fit1, '-r', label='fit degree 1')
    plt.plot(x_values, y_fit2, '-g', label='fit degree 2')
    plt.plot(x_values, y_fit3, '-b', label='fit degree 3')
    plt.plot(x_values, y_fit4, '-y', label='fit degree 4')
    plt.bar(bincenter, y_val, width=0.08, color='grey', yerr=yerr, label='hist')
    plt.xlabel('value')
    plt.ylabel('amount of data')
    plt.title('Exercise 3b) fitted measurements')
    plt.ylim(1200, 1800)
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('3b_fitted_plot.png')
    plt.clf()
    print('ex3b executed.')
    perr1 = np.sqrt(np.diag(pcov1))
    perr2 = np.sqrt(np.diag(pcov2))
    perr3 = np.sqrt(np.diag(pcov3))
    perr4 = np.sqrt(np.diag(pcov4))
    print(f'fit of degree 1: a = {popt1[0]:.2f} +/- {perr1[0]:.2f}, b = {popt1[1]:.2f} +/- {perr1[1]:.2f}')
    print(
        f'fit of degree 2: a = {popt2[0]:.2f} +/- {perr2[0]:.2f}, b = {popt2[1]:.2f} +/- {perr2[1]:.2f}, c = {popt2[2]:.2f} +/- {perr2[2]:.2f}')
    print(
        f'fit of degree 3: a = {popt3[0]:.2f} +/- {perr3[0]:.2f}, b = {popt3[1]:.2f} +/- {perr3[1]:.2f}, c = {popt3[2]:.2f} +/- {perr3[2]:.2f}, d = {popt3[3]:.2f} +/- {perr3[3]:.2f}')
    print(
        f'fit of degree 4: a = {popt4[0]:.2f} +/- {perr4[0]:.2f}, b = {popt4[1]:.2f} +/- {perr4[1]:.2f}, c = {popt4[2]:.2f} +/- {perr4[2]:.2f}, d = {popt4[3]:.2f} +/- {perr4[3]:.2f}, e = {popt4[4]:.2f} +/- {perr4[4]:.2f}')
    print('ex3c executed.')
    ndf1 = 18
    ndf2 = 17
    ndf3 = 16
    ndf4 = 15

    def chi2(pop, y, err, func):
        chi = 0
        for j, i in enumerate(y):
            chi += ((i - func(bincenter[j], *pop)) ** 2 / err[j] ** 2)
        return chi

    chi2_fit1 = chi2(popt1, counts, yerr, fit1)
    chi2_fit2 = chi2(popt2, counts, yerr, fit2)
    chi2_fit3 = chi2(popt3, counts, yerr, fit3)
    chi2_fit4 = chi2(popt4, counts, yerr, fit4)
    gof1 = chi2_fit1 / ndf1
    gof2 = chi2_fit2 / ndf2
    gof3 = chi2_fit3 / ndf3
    gof4 = chi2_fit4 / ndf4
    cdf1 = scs.chi2.cdf(gof1 * ndf1, ndf1)
    cdf2 = scs.chi2.cdf(gof2 * ndf2, ndf2)
    cdf3 = scs.chi2.cdf(gof3 * ndf3, ndf3)
    cdf4 = scs.chi2.cdf(gof4 * ndf4, ndf4)
    plt.scatter([1, 2, 3, 4], [gof1, gof2, gof3, gof4])
    plt.title('Exercise 3d) chi2 and polynom degree')
    plt.ylabel('Chi2/ndf')
    plt.xlabel('degree of the polynom')
    plt.savefig('3d_chi2_polynom.png')
    plt.grid()
    plt.show()
    plt.clf()

    return None


if __name__ == '__main__':
    # You can uncomment the exercises that you don't want to run. Here we have just one,
    # but in general you can have more.
    # ex_1()
    # ex2()
    ex3()
