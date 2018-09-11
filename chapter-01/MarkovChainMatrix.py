import numpy as np

class MarkovChain(object):
    def __init__(self, transition_matrix, states):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        transition_matrix: 2-D array
            A 2-D array representing the probabilities of change of 
            state in the Markov Chain.

        states: 1-D array 
            An array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in 
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}

    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.

        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return self.state_dict[np.random.choice(
            self.states, 
            p=self.transition_matrix[self.index_dict[current_state], :])]

    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: str
            The state of the current random variable.

        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

    def is_accessible(self, i_state, f_state):
        """
        Check if state f_state is accessible from i_state.

        Parameters
        ----------
        i_state: str
	    The state from which the accessibility needs to be checked.
        
        f_state: str
	    The state to which accessibility needs to be checked.
        """
        reachable_states = [i_state]
        for state in reachable_states:
	    if state == self.index_dict[f_state]:
	        return True
	    else:
	        reachable_states.append(np.nonzero(
	          self.transition_matrix[self.index_dict[i_state], :])[0])
        return False

    def is_irreducible(self):
        """
        Check if the Markov Chain is irreducible.
        """
        for (i, j) in combinations(self.states, self.states):
            if not self.is_accessible(i, j):
                return False
        return True

    def get_period(self, state):
        """
        Returns the period of the state in the Markov Chain.

        Parameters
        ----------
        state: str
            The state for which the period needs to be computed.
        """
        return gcd([len(i) for i in all_possible_paths])

    def is_aperiodic(self):
        """
        Checks if the Markov Chain is aperiodic. 
        """
        periods = [self.get_period(state) for state in self.states]
        for period in periods:
            if period != 1:
                return False
        return True

    def is_transient(self, state):
        """
        Checks if a state is transient or not.

        Parameters
        ----------
        state: str
            The state for which the transient property needs to be checked.
        """
        if all(self.transition_matrix[~self.index_dict[state], self.index_dict[state]] == 0):
            return True
        else:
            return False

    def is_absorbing(self, state):
     """
     Checks if the given state is absorbing.

     Parameters
     ----------
     state: str
     The state for which we need to check whether it's absorbing
     or not.
     """
     state_index = self.index_dict[state]
     if self.transition_matrix[state_index, state_index]
