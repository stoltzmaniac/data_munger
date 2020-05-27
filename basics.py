

def mean(data):
    """
    Creates arithmetic mean for function
    :param data: list of numbers
    :return: single float number
    """
    data_mean = sum(data)/len(data)
    return data_mean


def variance(data):
    """
    Return variance of sequence data.
    :param data: list of numbers
    :return: single float number
    """
    data_mean = mean(data)
    ss = sum((i-data_mean)**2 for i in data)
    return ss


def std_dev(data, deg_of_freedom=1):
    """
    Calculates the population standard deviation
    :param data:
    :param deg_of_freedom: Degrees of freedom, set as 0 to compute without sample
    :return: single float number
    """
    ss = variance(data)
    pvar = ss/(len(data)-deg_of_freedom)
    sd = pvar ** 0.5
    return sd


def covariance(data_x, data_y):
    """
    Calculates the covariance between x and y
    :param data_x: list of predictors
    :param data_y: list of targets
    :return: single float number
    """
    covar = 0.0
    x_mean = mean(data_x)
    y_mean = mean(data_y)
    for i in range(len(data_x)):
        covar += (data_x[i] - x_mean) * (data_y[i] - y_mean)
    return covar
