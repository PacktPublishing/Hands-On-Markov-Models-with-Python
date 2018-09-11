import numpy as np


class MultinomialHMM:
    def __init__(self, num_states, observation_states, prior_probabilities,
        transition_matrix, emission_probabilities):
        """
         Initialize Hidden Markov Model
         Parameters
         -----------
         num_states: int
         Number of states of latent variable
         observation_states: 1-D array
         An array representing the set of all observations
         prior_probabilities: 1-D array
         An array representing the prior probabilities of all the states
         of latent variable
         transition_matrix: 2-D array
         A matrix representing the transition probabilities of change of
         state of latent variable
         emission_probabilities: 2-D array
         A matrix representing the probability of a given observation
         given the state of the latent variable
         """
         # As latent variables form a Markov chain, we can use
         # use the previous defined MarkovChain class to create it
         self.latent_variable_markov_chain = MarkovChain(
            transition_matrix=transition_matrix,
            states=['z{index}'.format(index=index) for index in
                range(num_states)],
         )
         self.observation_states = observation_states
         self.prior_probabilities = np.atleast_1d(prior_probabilities)
         self.transition_matrix = np.atleast_2d(transition_matrix)
         self.emission_probabilities = np.atleast_2d(emission_probabilities)

    def observation_from_state(self, state):
     """
     Generate observation for a given state in accordance with
     the emission probabilities
     
     Parameters
     ----------
     state: int
        Index of the current state
     """
     state_index = self.latent_variable_markov_chain.index_dict[state]
     return np.random.choice(self.observation_states,
                             p=self.emission_probabilities[state_index, :])

    def generate_samples(self, no=10):
     """
     Generate samples from the hidden Markov model

     Parameters
     ----------
     no: int
        Number of samples to be drawn
     
     Returns
     -------
     observations: 1-D array
        An array of sequence of observations
     state_sequence: 1-D array
        An array of sequence of states
     """
     observations = []
     state_sequence = []
     initial_state = np.random.choice(
                    self.latent_variable_markov_chain.states,
                    p=self.prior_probabilities)
     state_sequence.append(initial_state)
     observations.append(self.observation_from_state(initial_state))
     current_state = initial_state
     for i in range(2, no):
         next_state = self.latent_variable_markov_chain.next_state(current_state)
         state_sequence.append(next_state)
         observations.append(self.observation_from_state(next_state))
         current_state = next_state
     return observations, state_sequence
