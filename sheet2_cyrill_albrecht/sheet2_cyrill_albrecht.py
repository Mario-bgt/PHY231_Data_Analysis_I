import numpy as np
import matplotlib.pyplot as plt


def mean(x):
    """Calculate the mean for an array-like object x.

    Parameters
    ----------
    x : array-like
        Array-like object containing the data.

    Returns
    -------
    mean : float
        The mean of the data.
    """
    # here goes your code
    mean_x = 0
    for i in x:
        mean_x = mean_x + i
    mean_x = mean_x / len(x)
    return mean_x


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    # here goes your code
    std = 0
    for i in x:
        std = std + (i - mean(x)) ** 2
    std = np.sqrt((1 / (len(x) - 1)) * std)
    return std


def variance(x):
    """Calculate the variance for an array-like object x."""
    # here goes your code
    var = std(x) ** 2
    return var


def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    # here goes your code
    mean_uncertainty = std(x) / (np.sqrt(len(x)))
    return mean_uncertainty


def covariance(x, y):
    covar = 0
    for i in range(len(x)):
        covar = covar + (x[i] - mean(x)) * (y[i] - mean(y))
    return covar / len(x)


def correlation(x, y):
    return covariance(x, y) / (std(x) * std(y))


def ex1():
    data = np.loadtxt("ironman.txt")
    age = 2010 - data[:, 1]
    time = data[:, 2]
    # a)
    mean_age = mean(age)
    mean_age_uncertainty = mean_uncertainty(age)
    variance_age = variance(age)
    std_age = std(age)
    # .2f means that the number is printed with two decimals. Check if that makes sense
    print(f"The mean age of the participants is {mean_age:.2f} +/- {mean_age_uncertainty:.2f} years. "
          f"The variance is {variance_age:.2f} and the standard deviation is {std_age:.2f}")
    mean_time = mean(time)
    mean_time_uncertainty = mean_uncertainty(time)
    variance_time = variance(time)
    std_time = std(time)
    time_under_35 = []
    time_over_35 = []
    print(f"The mean time of the participants is {mean_time:.2f} +/- {mean_time_uncertainty:.2f} minutes. "
          f"The variance is {variance_time:.2f} and the standard deviation is {std_time:.2f}")
    for row in data:
        if 2010 - row[1] < 35:
            time_under_35.append(row[2])
        else:
            time_over_35.append(row[2])
    average_time_under_35 = mean(time_under_35)
    average_time_over_35 = mean(time_over_35)
    mean_time_uncertainty_under_35 = mean_uncertainty(time_under_35)
    mean_time_uncertainty_over_35 = mean_uncertainty(time_over_35)
    print(f'The average total time for people under 35 years is {average_time_under_35:.0f} +/- '
          f'{mean_time_uncertainty_under_35:.2f} minutes and the average total time for people over 35 is'
          f' {average_time_over_35:.0f} +/- {mean_time_uncertainty_over_35:.2f} minutes.')
    print('Based on the data I can conclude that the people under 35 are faster.')

    def make_hist(lyst, amount_of_bars, ylabel, xlabel, file):
        entries = []
        error = []
        x_pos = np.arange(amount_of_bars - 1)
        counter = np.linspace(min(lyst), max(lyst), amount_of_bars)
        for i in np.arange(0, amount_of_bars - 1):
            L = []
            for x in lyst:
                if counter[i] <= x <= counter[i + 1]:
                    L.append(x)
            list_mean = len(L)
            list_std = np.sqrt(list_mean)
            entries.append(list_mean)
            error.append(list_std)
        fig, ax = plt.subplots()
        ax.bar(x_pos, entries, yerr=error)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(np.round(np.linspace(min(lyst), max(lyst), amount_of_bars - 1), 1))
        plt.savefig(file)
        plt.show()

    make_hist(age, 13, 'amount of people', 'age of the participants', 'age_hist.png')
    make_hist(time, 12, 'amount of people', 'Total time taken in minutes', 'time_hist.png')

    total_Rank = data[:, 0]
    swimming_time = data[:, 3]
    cycling_time = data[:, 5]
    running_time = data[:, 7]
    time_in_sec = 60 * time
    covariance_rank_time = covariance(total_Rank, time)
    covariance_age_time = covariance(age, time)
    covariance_time_swimming = covariance(time, swimming_time)
    covariance_cycling_running = covariance(cycling_time, running_time)
    correlation_rank_time = correlation(total_Rank, time)
    correlation_age_time = correlation(age, time)
    correlation_time_swimming = correlation(time, swimming_time)
    correlation_cycling_running = correlation(cycling_time, running_time)
    print(
        f'Covariance between total Rank and total time: {covariance_rank_time:.2f} with the correlation of {correlation_rank_time:.5f}')
    print(f'The scatter plot shows a line with a almost no spread, this indicates a high covariance and a high '
          f'correlation.')
    print(
        f'Covariance between age and total time: {covariance_age_time:.2f} with the correlation of {correlation_age_time:.5f}')
    print(f'The scatter plot shows a big spread, this indicates a low covariance and a low '
          f'correlation')
    print(
        f'Covariance between total time and swimming time: {covariance_time_swimming:.2f} with the correlation of {correlation_time_swimming:.5f}')
    print(f'The scatter plot shows a big spread, this indicates a low covariance and a low '
          f'correlation')
    print(
        f'Covariance between cycling time and running time: {covariance_cycling_running:.2f} with the correlation of {correlation_cycling_running:.5f}')
    covariance_age_time_in_sec = covariance(time_in_sec, age)
    print(f'The scatter plot shows a big spread, this indicates a low covariance and a low '
          f'correlation')
    correlation_age_time_in_sec = correlation(time_in_sec, age)
    print(
        f'Covariance between age and total time in seconds: {covariance_age_time_in_sec:.2f} with the correlation of {correlation_age_time_in_sec:.5f}. The correlation stays the same')
    print(f'The scatter plot shows a big spread, this indicates a low covariance and a low '
          f'correlation')


def ex2():
    radiation = np.loadtxt("radiation.txt")
    mean_x = 0
    for i, j in radiation:
        mean_x = mean_x + ((1 / (j ** 2)) * i)
    weight = 0
    for i in radiation[:, 1]:
        weight = weight + (1 / (i ** 2))
    mean_x = (mean_x / weight) * 8760
    uncertainty = (1 / (weight) ** (1 / 2)) * 8760
    print(f'Average radiation level: {mean_x:.4f} +/- {uncertainty:.5f} mSv/year.')
    print('The evaluated radiation is just slightly higher than the reference value. Based on this little difference '
          'the radiation level is compatible')


if __name__ == '__main__':
    ex1()
    ex2()  # uncomment to run ex
