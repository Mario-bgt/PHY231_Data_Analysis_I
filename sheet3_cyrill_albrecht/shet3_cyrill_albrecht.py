import math

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


def pmf_binomial(r, n, p):
    binomial = (np.math.factorial(n) / (np.math.factorial(n - r) * np.math.factorial(r)) * (
            (p ** r) * ((1 - p) ** (n - r))))
    return binomial


def ex1():
    print("Exercise 1")

    # Exercise1a
    def probability_distribution(n):
        n = 4
        p = 0.85
        all_r = list(range(n + 1))
        dist = []
        for r in all_r:
            dist.append(pmf_binomial(r, n, p))
        plt.figure()
        plt.bar(all_r, dist)
        plt.ylabel("probability getting detected")
        plt.xlabel("Amount of detectors")
        plt.savefig('ex1_a.pdf')
        plt.show()
    probability_distribution(4)

    # Exercise 1b

    n = 2
    probability = 0
    while probability <= 0.99:
        p = 0.85
        n = n + 1
        all_r = list(range(n + 1))
        dist = []
        for r in all_r:
            dist.append(pmf_binomial(r, n, p))
        probability = sum(dist[3:])
    print(f"{n} detectors are needed that the particle detection efficiency is above 99%")

    # Exercise 1c

    n = 4
    p = 0.85
    all_r = list(range(n + 1))
    dist = []
    for r in all_r:
        dist.append(pmf_binomial(r, n, p))
    p = dist[4]
    n = 1000
    dist = []
    all_r = list(range(n + 1))
    for r in all_r:
        dist.append(pmf_binomial(r, n, p))
    plt.figure()
    plt.bar(all_r, dist)
    plt.xlabel("Number of particles detected")
    plt.ylabel("Probability Density")
    plt.savefig('ex1_c.pdf')
    plt.show()
    print("The square root of 1000 is approximately the same as the width, what tells that it agrees with Poisson")


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

    # Exercise 4a

    n = 500
    p = 0.82
    all_r = list(range(n + 1))
    dist = []
    for r in all_r:
        dist.append(pmf_binomial(r, n, p))
    probability = sum(dist[390:])
    print(f"The probability to detect more than 390 Z-Bosons is {probability:.4f}")

    # Exercise 4b

    mean = n * p
    variance = n * p * (1 - p)
    x_axis = list(range(n + 1))
    all_r = list(range(n + 1))
    y = []
    for val in x_axis:
        y.append((1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((val - mean) ** 2) / (2 * variance)))
    plt.figure()
    plt.plot(x_axis, y, 'g')
    plt.bar(all_r, dist)
    plt.xlabel("Number of particles detected")
    plt.ylabel("Probability Density")
    plt.savefig('ex4b.pdf')
    plt.show()
    print("The gaussian approximation fits well")

    # Exercise 4c

    n = 500
    mean = n * p
    variance = n * p * (1 - p)
    standard_deviation = np.sqrt(variance)
    x_1 = list(range(n + 1))
    x_2 = list(range(n))
    y_1 = []
    for val in x_1:
        y_1.append((1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((val - mean) ** 2) / (2 * variance)))
    lamda = n * 0.82
    y_2 = []
    for val in x_2:
        i = 1
        for r in range(1, val):
            i = i * (lamda / r)
        y_2.append(math.pow(np.e, -lamda) * i)
    plt.figure()
    plt.plot(x_1, y_1, 'g')
    plt.plot(x_2, y_2)
    plt.ylabel("Probability Density")
    plt.xlabel("Detected particles")
    plt.title("Poisson Distribution vs Gaussian")
    plt.savefig("ex4c.pdf")
    plt.show()
    print("it's a good approximation, even if the variance in the poisson is bigger (as expected)")

    # Exercise 4d

    n = 500
    time = 125
    probability = 0.18
    n_2 = n / time
    mean = n_2 * probability
    variance = n_2 * probability * (1 - probability)
    x_1 = list(range(int(n_2 + 1)))
    x_2 = list(range(int(n_2)))
    y_1 = []
    for val in x_1:
        y_1.append((1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((val - mean) ** 2) / (2 * variance)))
    y_2 = []
    for val in x_2:
        i = 1
        for r in range(1, val):
            i = i * (mean / r)
        y_2.append(math.pow(np.e, -mean) * i)
    plt.figure()
    plt.plot(x_1, y_1, 'g')
    plt.plot(x_2, y_2)
    plt.ylabel("Probability Density")
    plt.xlabel("Detected particles")
    plt.savefig("ex4_d.pdf")
    plt.show()
    print("Now it isn't a good approximation, because we have less data")


if __name__ == '__main__':
    ex1()
    ex3()  # uncomment to run ex3
    ex4()  # uncomment to run ex4
