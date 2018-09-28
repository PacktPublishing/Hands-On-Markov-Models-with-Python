import numpy as np


def coin_mle(data):
    """
    Returns the learned probability of getting a heads using MLE.
    
    Parameters
    ----------
    data: list, array-like
        The list of observations. 1 for heads and 0 for tails.
     
    Returns
    -------
    theta: The learned probability of getting a heads.
    """
    data = np.array(data)
    n_heads = np.sum(data)
    return n_heads / data.size
