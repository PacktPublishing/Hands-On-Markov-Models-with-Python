def weather_fit(data):
    """
    Learn the transition and emission probabilities from the given data
    for the weather model.

    Parameters
    ----------
    data: 2-D list (array-like)
        Each data point should be a tuple of size 2 with the first element
        representing the state of Weather and the second element representing
        whether it rained or not.

        Sunny = 0, Cloudy = 1, Windy = 2
        Rain = 0, No Rain = 1

    Returns
    -------
    transition probability: 2-D array
        The conditional distribution respresenting the transition probability
        of the model.

    emission probability: 2-D array
        The conditional distribution respresenting the emission probability
        of the model.
    """
    data = np.array(data)
    transition_counts = np.zeros((3, 3))
    emission_counts = np.zeros((3, 2))
    for index, datapoint in enumerate(data):
        if index != len(data)-1:
            transition_counts[data[index][0], data[index+1][0]] += 1
        emission_counts[data[index][0], data[index][1]] += 1
    transition_prob = transition_counts / np.sum(transition_counts, axis=0)
    emission_prob = (emission_counts.T / np.sum(emission_counts.T,
                                                axis=0)).T
    return transition_prob, emission_prob
