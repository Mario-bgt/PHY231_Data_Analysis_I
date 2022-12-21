"""Datenanalysis sheet 4 of Mario Baumgartner"""
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
import math


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
    plt.title('Exersice 1a) negative log likelihood plot')
    plt.xlabel('alpha')
    plt.ylabel('NLL(alpha)')
    plt.grid()
    # plt.show()
    plt.savefig('ex1a.png')
    plt.clf()
    print("ex1a executed.")
    # Part b)
    nll_min = x_val[y_val.index(min(y_val))]
    print(f'1b) The maximum likelihood estimator is {nll_min:.2f}, which one can also see by looking at the plot.')
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
    plt.title('Exercise 2a) twice negative log likelihood plot with shift to 0')
    plt.xlabel('tau')
    plt.ylabel('2*NLL(tau)')
    plt.grid()
    # plt.show()
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
    # plt.show()
    plt.savefig('ex2b.png')
    plt.clf()
    print(f'2b) No it does not make a difference since the two lines are very close to one-other.')
    print('ex2b executed.')

    # Part c)
    def ex2chi2(t, y, x):
        chi2 = 0
        for i, y in enumerate(y):
            chi2 += (y / np.sqrt(y) - (125 * np.exp(-x[i] / t) / (t * (1 - np.exp(-5 / t)))) / np.sqrt(y))**2
        return chi2

    counts, edges = np.histogram(data, bins=40)
    bincenter = (edges[:-1] + edges[1:]) / 2
    y_chi2 = [ex2chi2(i, counts, bincenter) for i in x_val]
    y_chi2 = [y_chi2[i]-min(y_chi2) for i in range(1000)]
    plt.plot(x_val, y_chi2, color='green', label='Chi2')
    plt.plot(x_val, y_val, color='red', label='Binned 2NLL')
    plt.plot(x_val, y_binned, color='blue', label='2NLL unbinned')
    plt.xlabel('tau [nanoseconds]')
    plt.ylabel('amount of data')
    plt.legend()
    plt.title('Exercise 2c) added chi2 value')
    plt.grid()
    # plt.show()
    plt.savefig('ex2c.png')
    plt.clf()
    print(f"2c) As already observed in 2b) the binned-NLL and unbinnned-NLL graphs are nearly equal.\n But the chi2 "
          f"graph is off by 0.1. A reason could be that the data doesn't follow a gaussian distribution")
    print('ex2c executed.')
    # Part d)
    y_binned = [binned_2nll(data, prob_2, i, 2) for i in x_val]
    y_binned = [y_binned[i] - min(y_binned) for i in range(1000)]
    counts, edges = np.histogram(data, bins=2)
    bincenter = (edges[:-1] + edges[1:]) / 2
    y_chi2 = [ex2chi2(i, counts, bincenter) for i in x_val]
    y_chi2 = [y_chi2[i] - min(y_chi2) for i in range(1000)]
    plt.plot(x_val, y_chi2, color='green', label='Chi2')
    plt.plot(x_val, y_val, color='red', label='Binned 2NLL')
    plt.plot(x_val, y_binned, color='blue', label='2NLL unbinned')
    plt.xlabel('tau [nanoseconds]')
    plt.ylabel('amount of data')
    plt.legend()
    plt.title('Exercise 2d) with only two bins')
    plt.grid()
    # plt.show()
    plt.savefig('ex2d.png')
    plt.clf()
    print('2d) The data is now closer to a gaussian distribution thus resulting in the graphs being closer to each.\n'
          'Furthermore since the slope of chi2 and NLL binned are more flat it means their uncertainty is higher.\n'
          'This is as expected since we use a lot less information.')
    print('ex2d executed.')


def ex3():
    data = np.loadtxt('polynomial_data.txt')
    counts, edges = np.histogram(data, bins=20)
    bincenter = (edges[:-1] + edges[1:]) / 2
    y_val = counts
    yerr = [np.sqrt(i) for i in counts]
    plt.bar(bincenter, y_val, width=0.08, color='lightblue', yerr=yerr)
    plt.errorbar(bincenter, y_val, yerr=yerr, color='blue', drawstyle='steps-mid')
    plt.xlabel('value')
    plt.ylabel('amount of data')
    plt.title('Exercise 3a) binned measurements')
    plt.ylim(1200, 1800)
    plt.grid()
    # plt.show()
    plt.savefig('ex3a.png')
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
    y_fit1 = [fit1(i, *popt1) for i in x_values]
    y_fit2 = [fit2(i, *popt2) for i in x_values]
    y_fit3 = [fit3(i, *popt3) for i in x_values]
    y_fit4 = [fit4(i, *popt4) for i in x_values]
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
    # plt.show()
    plt.savefig('ex3b.png')
    plt.clf()
    print('ex3b executed.')
    perr1 = np.sqrt(np.diag(pcov1))
    perr2 = np.sqrt(np.diag(pcov2))
    perr3 = np.sqrt(np.diag(pcov3))
    perr4 = np.sqrt(np.diag(pcov4))
    print(f'fit of degree 1: a = {popt1[0]:.2f} +/- {perr1[0]:.2f}, b = {popt1[1]:.2f} +/- {perr1[1]:.2f}')
    print(f'fit of degree 2: a = {popt2[0]:.2f} +/- {perr2[0]:.2f}, b = {popt2[1]:.2f} +/- {perr2[1]:.2f}, '
          f'c = {popt2[2]:.2f} +/- {perr2[2]:.2f}')
    print(f'fit of degree 3: a = {popt3[0]:.2f} +/- {perr3[0]:.2f}, b = {popt3[1]:.2f} +/- {perr3[1]:.2f},'
          f' c = {popt3[2]:.2f} +/- {perr3[2]:.2f}, d = {popt3[3]:.2f} +/- {perr3[3]:.2f}')
    print(f'fit of degree 4: a = {popt4[0]:.2f} +/- {perr4[0]:.2f}, b = {popt4[1]:.2f} +/- {perr4[1]:.2f},'
          f' c = {popt4[2]:.2f} +/- {perr4[2]:.2f}, d = {popt4[3]:.2f} +/- {perr4[3]:.2f},'
          f' e = {popt4[4]:.2f} +/- {perr4[4]:.2f}')
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
    plt.scatter([1, 2, 3, 4], [gof1, gof2, gof3, gof4])
    plt.title('Exercise 3d) chi2 and polynom degree')
    plt.ylabel('Chi2/ndf')
    plt.xlabel('degree of the polynom')
    plt.savefig('ex3d.png')
    plt.grid()
    # plt.show()
    plt.clf()
    print('ex3d executed.')
    # Part e)
    print(f'I would argue it was a Polynom of degree 3 since the goodness of fit is the best and also the match of '
          f'degree 3 and 4 is basically equally good.')
    print('ex3e executed.')
    return None


if __name__ == '__main__':
    ex_1()
    ex2()
    ex3()
