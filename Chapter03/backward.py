def backward(obs, transition, emission, init):
    """
    Runs backward algorithm on the HMM.

    Parameters
    ----------
    obs:        1D list, array-like
                The list of observed states.

    transition: 2D array-like
                The transition probability of the HMM.
                size = {n_states x n_states}

    emission:   1D array-like
                The emission probabiltiy of the HMM.
                size = {n_states}

    init:       1D array-like
                The initial probability of HMM.
                size = {n_states}

    Returns
    -------
    float: Probability value for the obs to occur.
    """
    n_states = transition.shape[0]
    bkw = [{} for t in range(len(obs))]
    T = len(obs)
    
    for y in range(n_states):
        bkw[T-1][y] = 1
    for t in reversed(range(T-1)):
        for y in range(n_states):
            bkw[t][y] = sum((bkw[t+1][y1] * transition[y][y1] * emission[obs[t+1]]) for y1 in 
                                    range(n_states))
    prob = sum((init[y] * emission[obs[0]] * bkw[0][y]) for y in range(n_states))
    return prob
