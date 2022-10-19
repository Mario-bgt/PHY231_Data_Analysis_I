"""Datenanalysis sheet 3 of Mario Baumgartner"""
# TO DO: -delet show, enable all ex, ans ex1c, make ex2


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


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
        text
    n: int
        text
    p: float

    """
    number_of_ways = np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r))
    r_succ_n_r_fail = (p ** r) * ((1 - p) ** (n - r))
    return number_of_ways * r_succ_n_r_fail


def ex1():
    print("Exercise 1")

    ####Ex 1 a) ####
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
        plt.show()

    plot_probability_distribution(4)

    ####Ex 1 b) ####
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

    ####Ex 1 c) ####
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
    plt.savefig('ex1_c_probability_distribution.pdf')
    plt.show()


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
    ####Ex 4 a) ####
    n = 500
    p = 0.82
    r_values = list(range(n + 1))
    dist = [binom_pmf(r, n, p) for r in r_values]
    res = sum(dist[390:])
    print(f"The chance of detecting more than 390 Particles is {res}")

    ###Ex 4 b) ####
    mean = n * p
    var = n * p * (1 - p)
    std = np.sqrt(var)
    print(mean, var, std)

    def pdf(x, mean, var):
        return (1 / np.sqrt(2 * np.pi*var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    x = np.linspace(0, n, n + 1)
    plt.figure()
    plt.plot(x, pdf(x, mean, var), 'r')
    plt.bar(r_values, dist)
    plt.xlabel("Number of particles detected")
    plt.savefig('ex4_b_probability_distribution.pdf')
    plt.show()



if __name__ == '__main__':
    # ex1()
    # ex3()  # uncomment to run ex3
    ex4()  # uncomment to run ex4
