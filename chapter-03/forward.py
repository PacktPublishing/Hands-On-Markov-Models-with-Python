import numpy as np

transition_matrix = 
    np.array([[0.33, 0.33,    0,    0,    0, 0.33,    0,    0,    0,    0,    0,    0,    0],
              [0.33, 0.33, 0.33,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
              [   0, 0.25, 0.25, 0.25,    0,    0, 0.25,    0,    0,    0,    0,    0,    0],
              [   0,    0, 0.33, 0.33, 0.33,    0,    0,    0,    0,    0,    0,    0,    0],
              [   0,    0,    0, 0.33, 0.33,    0,    0, 0.33,    0,    0,    0,    0,    0],
              [0.33,    0,    0,    0,    0, 0.33,    0,    0, 0.33,    0,    0,    0,    0],
              [   0,    0, 0.33,    0,    0,    0, 0.33,    0,    0,    0, 0.33,    0,    0],
              [   0,    0,    0,    0, 0.33,    0,    0, 0.33,    0,    0,    0,    0, 0.33],
              [   0,    0,    0,    0,    0, 0.33,    0,    0, 0.33, 0.33,    0,    0,    0],
              [   0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33,    0,    0],
              [   0,    0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33,    0],
              [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 0.33, 0.33, 0.33],
              [   0,    0,    0,    0,    0,    0,    0, 0.33,    0,    0,    0, 0.33, 0.33]])

emission = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

init_prob = np.array([0.077, 0.077, 0.077, 0.077, 0.077, 0.077, 0.077,
                      0.077, 0.077, 0.077, 0.077, 0.077, 0.077])

def forward(obs, transition, emission, init):
    """
    Runs forward algorithm on the HMM.

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
    fwd = [{}]

    for i in range(n_states):
        fwd[0][y] = init[i] * emission[obs[0]]
    for t in range(1, len(obs)):
        fwd.append({})
        for i in range(n_states):
            fwd[t][i] = sum((fwd[t-1][y0] * transition[y0][i] * emission[obs[t]]) for y0 in 
                                    range(n_states))
    prob = sum((fwd[len(obs) - 1][s]) for s in range(n_states))
    return prob
