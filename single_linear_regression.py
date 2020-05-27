from basics import mean, variance, covariance


def fit_single_linear_regression(x, y):
    """
    Fits a single linear regression model in the format: y = m*x + b
    :param x: list of predictor values
    :param y: list of target values
    :return: dict of {'m': float, 'b': float} where m is the slope and b is the y-intercept
    """
    m = covariance(data_x=x, data_y=y) / variance(data=x)
    b = mean(y) - m * mean(x)
    coefficients = {
        'm': m,
        'b': b
    }
    return coefficients


def predict_single_linear_regression(x, model):
    """
    Utilizes formula y = m*x + b where we are given only coefficients and x
    :param x: list of values to predict
    :param model: dict of results of fit_single_linear_regression
    :return: list of predicted outputs
    """
    m = model.get('m')
    b = model.get('b')
    predicted_y = [m*i + b for i in x]
    return predicted_y


# TODO: create a way to evaluate the model (i.e. R squared)
