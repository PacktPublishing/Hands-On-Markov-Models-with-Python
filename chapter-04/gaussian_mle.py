import numpy as np


def gaussian_mle(data):
    """
    Returns the learned parameters of the Normal Distribution using MLE.

    Parameters
    ----------
    data: list, array-like
        The list of observed variables.

    Returns
    -------
    \mu: The learned mean of the Normal Distribution.
    \sigma: The learned standard deviation of the Normal Distribution.
    """
    data = np.array(data)
    mu = np.mean(data)
    variance = np.sqrt(np.mean((data - mu)**2))
    return mu, variance
