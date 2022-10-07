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
    N = len(x)
    x_mean = 0
    for i in x:
        x_mean += i
    x_mean = (1 / N) * x_mean
    return x_mean


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    x_mean = mean(x)
    N = len(x)
    std = 0
    for i in x:
        std += (i - x_mean) ** 2
    std = np.sqrt((1 / (N - 1)) * std)
    return std


def variance(x):
    """Calculate the variance for an array-like object x."""
    x_mean = mean(x)
    N = len(x)
    var = 0
    for i in x:
        var += (i - x_mean) ** 2
    var = (1 / N) * var
    return var


def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    N = len(x)
    return std(x) / (np.sqrt(N))


def covariance(x, y):
    """Calculate the covariance of two array-like objects x and y"""
    cov = 0
    for i in range(len(x)):
        cov += (x[i] - mean(x)) * (y[i] - mean(y))
    return cov / len(x)


def correlation(x, y):
    """Calculate the correlation of two array-like objects x and y"""
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
    # .2f means that the number is printed with two decimals. Check if that makes sense --> it does make sense
    print(f"The mean age of the participants is {mean_age:.2f} +/- {mean_age_uncertainty:.2f} years. Where as the "
          f"variance is {variance_age:.2f} years and the standard deviation  is {std_age:.2f} years")
    mean_time = mean(time)
    mean_time_uncertainty = mean_uncertainty(time)
    variance_time = variance(time)
    std_time = std(time)
    print(f"The mean time of the participants is {mean_time:.2f} +/- {mean_time_uncertainty:.2f} minutes. Where as the "
          f"variance is {variance_time:.2f} min and the standard deviation  is {std_time:.2f} min")
    # b)
    time_for_age_under_35 = []
    time_for_age_above_35 = []
    for row in data:
        if 2010 - row[1] < 35:
            time_for_age_under_35.append(row[2])
        else:
            time_for_age_above_35.append(row[2])
    mean_time_for_age_under_35 = mean(time_for_age_under_35)
    mean_time_for_age_above_35 = mean(time_for_age_above_35)
    mean_time_for_age_under_35_uncertainty = mean_uncertainty(time_for_age_under_35)
    mean_time_for_age_above_35_uncertainty = mean_uncertainty(time_for_age_above_35)
    print(f'The average of the total time for people younger than 35 years is {mean_time_for_age_under_35:.0f} +/- '
          f'{mean_time_for_age_under_35_uncertainty: .2f} minutes, while the average for people above 35 is '
          f'{mean_time_for_age_above_35:.0f} +/- {mean_time_for_age_above_35_uncertainty:.2f} minutes.')
    print(f'I can conclude with this data that the group with people below the age of 35 is faster.')

    # c)

    def make_hist(lyst, bar_amount, ylabel, xlabel, file):
        CTEs = []  # will be filled with the amount of entries in each bin
        error = []  # will be filled with the error upon each bin
        x_pos = np.arange(bar_amount - 1)  # is the amount of bars in the plot
        counter = np.linspace(min(lyst), max(lyst), bar_amount)
        for i in np.arange(0, bar_amount - 1):
            temp_lyst = [x for x in lyst if counter[i] <= x <= counter[i + 1]]
            temp_lyst_mean = len(temp_lyst)
            temp_lyst_std = np.sqrt(temp_lyst_mean)
            CTEs.append(temp_lyst_mean)
            error.append(temp_lyst_std)
        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(np.round(np.linspace(min(lyst), max(lyst), bar_amount - 1), 1))
        ax.yaxis.grid(True)
        # Save the figure and show
        plt.tight_layout()
        plt.savefig(file)
        # plt.show()

    make_hist(time, 20, 'amount of people', 'Total time taken in minutes', 'time_hist.png')
    make_hist(age, 20, 'amount of people', 'age of the participants', 'age_hist.png')
    # d)
    # I didn't understand how to do this exactly
    # e)
    rank = data[:, 0]
    swimming_time = data[:, 3]
    cycling_time = data[:, 5]
    running_time = data[:, 7]
    cov_rank_time = covariance(rank, time)
    cor_rank_time = correlation(rank, time)
    cov_age_time = covariance(age, time)
    cor_age_time = correlation(age, time)
    cov_time_swimming = covariance(time, swimming_time)
    cor_time_swimming = correlation(time, swimming_time)
    cov_cycling_running = covariance(cycling_time, running_time)
    cor_cycling_running = correlation(cycling_time, running_time)
    print(f'The covariance between the total rank and the total time is: {cov_rank_time: .2f}, whereas the '
          f'correlation is: {cor_rank_time: .5f}.')
    print('Looking at the scatter plot implies a high value for the correlation and covariance, since all points are '
          'close to one line and spread across the plot, as also obtained here by calculation.')
    print(f'The covariance between the age and the total time in minutes is: {cov_age_time: .2f}, whereas the '
          f'correlation is: {cor_age_time: .5f}')
    print('Looking at the scatter plot implies a low value for the correlation since the values are spread. And since'
          'they are all spread around one point the covariance is as expected low.')
    print(f'The covariance between the total time and swimming time is: {cov_time_swimming: .2f}, whereas the '
          f'correlation is: {cor_time_swimming: .5f}')
    print(f'The covariance between the cycling time and the running time is: {cov_cycling_running: .2f}, whereas the '
          f'correlation is: {cor_cycling_running: .5f}')
    time_in_sec = 60*data[:, 2]
    cov_age_time_in_sec = covariance(time_in_sec, age)
    cor_age_time_in_sec = correlation(time_in_sec, age)
    print(f'The covariance between the age and the total time in seconds is: {cov_age_time_in_sec: .2f}, whereas the '
          f'correlation is: {cor_age_time_in_sec: .5f} which as expected is still the same as with time in minutes..')


def ex2():
    radiation = np.loadtxt("radiation.txt")
    x_mean = 0
    weight = 0
    for value, error in radiation:
        x_mean += (1/(error**2))*value
        weight += 1/(error**2)
    x_mean = x_mean/weight
    x_mean = x_mean*24*365
    uncertainty = 1/np.sqrt(weight)
    uncertainty = uncertainty*24*365
    print(f'The average radiation level is {x_mean: .3f} +/-{uncertainty: .6f}, in mSv per year')
    print('Since the natural background radiation is measured to be 2.4 mSv/y which is only a bit lower than the')
    print('obtained value for the radiation in the room, I conclude the room is compatible with the natural background '
          'radiation')


if __name__ == '__main__':
    ex1()
    # ex2()  # uncomment to run ex2
