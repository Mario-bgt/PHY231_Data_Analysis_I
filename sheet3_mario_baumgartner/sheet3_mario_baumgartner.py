"""Datenanalysis sheet 3 of Mario Baumgartner"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math


def integrate(dist, lower, upper):
    """Integrate the pdf of a distribution between lower and upper.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        A scipy.stats distribution object.
    lower : float
        Lower limit of the integration.
    upper : float
        Upper limit of the integration.

    Returns
    -------
    integral : float
        The integral of the pdf between lower and upper.
    """
    return dist.cdf(upper) - dist.cdf(lower)


def binom_pmf(r, n, p):
    """Calculate the Binomial Distribution
    Parameters
    ---------
    r: int
        number of successes
    n: int
        successive independent trials
    p: float
        probability of a success

     Returns
    -------
        the total probability of achieving r success and n-r failure

    """
    number_of_ways = np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))
    r_succ_n_r_fail = (p ** r) * ((1 - p) ** (n - r))
    return number_of_ways * r_succ_n_r_fail


def ex1():
    print("Exercise 1")

    # Ex 1 a)
    def plot_probability_distribution(n):
        n = 4
        p = 0.85
        r_values = list(range(n + 1))
        dist = [binom_pmf(r, n, p) for r in r_values]
        plt.figure()
        plt.bar(r_values, dist)
        plt.ylabel("Chance of getting detected")
        plt.xlabel("Amount of detectors")
        plt.savefig('ex1_a_probability_distribution.pdf')
        # plt.show()

    plot_probability_distribution(4)

    # Ex 1 b)
    n = 2
    while True:
        n += 1
        p = 0.85
        r_values = list(range(n + 1))
        dist = [binom_pmf(r, n, p) for r in r_values]
        prob = sum(dist[3:])
        if prob >= 0.99:
            break
    print(f"We need at least {n} detectors in order to ensure that the particle detection efficiency is above 99%")

    # Ex 1 c)
    n = 4
    p = 0.85
    r_values = list(range(n + 1))
    dist = [binom_pmf(r, n, p) for r in r_values]
    p = dist[4]
    n = 1000
    r_values = list(range(n + 1))
    dist = [binom_pmf(r, n, p) for r in r_values]
    plt.figure()
    plt.bar(r_values, dist)
    plt.xlabel("Number of particles detected")
    plt.ylabel("Probability Density")
    plt.savefig('ex1_c_probability_distribution.pdf')
    # plt.show()
    print("Since the width is approximately sqrt(1000) I would argue it agrees with one whom would be expected from a "
          "Poisson distribution")


def ex3():
    print("Exercise 3")
    norm_dist_shifted = scipy.stats.norm(loc=1, scale=0.01)
    prob_a = integrate(norm_dist_shifted, 0.97, 1.03)
    prob_b = integrate(norm_dist_shifted, 0.99, 1.00)
    prob_c = integrate(norm_dist_shifted, 0.95, 1.05)
    prob_d = integrate(norm_dist_shifted, 0, 1.015)
    print("Probabilities:")
    print(f"3a) {prob_a:.3f} to be within [0.97, 1.03]m")
    print(f"3b) {prob_b:.3f} to be within [0.99, 1.00]m")
    print(f"3c) {prob_c:.6f} to be within [0.95, 1.05]m")
    print(f"3d) {prob_d:.3f} to be less than 1.015 m")


def ex4():
    print("Exercise 4")
    # Ex 4 a)
    n = 500
    p = 0.82
    r_values = list(range(n + 1))
    dist = [binom_pmf(r, n, p) for r in r_values]
    res = sum(dist[390:])
    print(f"The chance of detecting more than 390 Particles is {res:.3f}")

    # Ex 4 b)
    mean = n * p
    var = n * p * (1 - p)
    std = np.sqrt(var)
    x = list(range(n + 1))
    y = []
    for val in x:
        y.append((1 / np.sqrt(2 * np.pi * var)) * np.exp(-((val - mean) ** 2) / (2 * var)))
    plt.figure()
    plt.plot(x, y, 'r')
    plt.bar(r_values, dist)
    plt.xlabel("Number of particles detected")
    plt.ylabel("Probability Density")
    plt.savefig('ex4_b_probability_distribution.pdf')
    # plt.show()
    print("The approximation fits very good")

    # Ex 4 c)
    n = 500
    mean = n * p
    var = n * p * (1 - p)
    std = np.sqrt(var)
    x1 = list(range(n + 1))
    y1 = [(1 / np.sqrt(2 * np.pi * var)) * np.exp(-((val - mean) ** 2) / (2 * var)) for val in x1]
    lamda = 410  # average number of successes
    x2 = list(range(n))
    # I couldn't think of a better way to calculate y values since python itself cant convert 200 or more factorial to
    # float. If you have a better idea please let me know.
    y2 = []
    for val in x2:
        temp = 1
        for r in range(1, val):
            temp = temp * (lamda / r)
        y2.append(math.pow(np.e, -lamda) * temp)
    # y2 = [(math.pow(np.e, -lamda) * lamda ** val) / (math.factorial(int(val))) for val in x2]
    plt.figure()
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2)
    plt.ylabel("Probability Density")
    plt.xlabel("Amount of detected particles")
    plt.title("Poisson Distribution vs Gaussian")
    plt.savefig("ex4_c_poisson_gauss.pdf")
    # plt.show()
    print(f"As expected the poisson distribution has a bigger variance but is a good approximation")

    # Ex 4 d)
    n = 500
    time = 125
    p = 0.18
    new_n = n/time
    mean = new_n * p
    var = new_n * p * (1 - p)
    x1 = list(range(int(new_n + 1)))
    y1 = [(1 / np.sqrt(2 * np.pi * var)) * np.exp(-((val - mean) ** 2) / (2 * var)) for val in x1]
    x2 = list(range(int(new_n)))
    y2 = []
    for val in x2:
        temp = 1
        for r in range(1, val):
            temp = temp * (mean / r)
        y2.append(math.pow(np.e, -mean) * temp)
    # y2 = [(math.pow(np.e, -lamda) * lamda ** val) / (math.factorial(int(val))) for val in x2]
    plt.figure()
    plt.plot(x1, y1, 'r')
    plt.plot(x2, y2)
    plt.ylabel("Probability Density")
    plt.xlabel("Amount of detected particles")
    plt.savefig("ex4_d_poisson_gauss.pdf")
    # plt.show()
    print("Since now we have way less Data the distribution doesnt look very smooth and isn't even close to part c)")


if __name__ == '__main__':
    ex1()
    ex3()
    ex4()
