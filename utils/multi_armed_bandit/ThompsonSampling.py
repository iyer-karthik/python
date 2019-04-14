# -*- coding: utf-8 -*-
# -*- author: Karthik Iyer -*-

# Disclaimer: .py file mostly copied from Cam Davidson Pilon's
# book "Bayesian Method for Hackers" (Chapter 6)

import numpy as np


class BanditSimulation(object):
    """
    Class to simulate rewards for a Multi armed bandit problem. 

    Given true hidden probabilities, the .pull() method will randomly 
    simulate a reward for the chosen arm.

    Parameters
    ----------
    prob_array : np.array
        numpy array for true (hidden) probabilities of expected rewards of 
        arms/ versions

    """

    def __init__(self, prob_array):
        self.prob_array = prob_array

        # Ideally always choose this arm
        self.optimal = np.argmax(np.array(prob_array))

        # Define a rand object and freeze it. Only needed once while instantiaing
        # BanditSimulation object
        self._rand = np.random.rand

    def get_reward(self, arm):
        """
        arm is which arm to pull. Simulates either a hit or a miss

        Parameters
        ----------
        arm : int
            index for which arm to pull and simulate a reward (hit/miss). Note 
            that larger the value of prob_array[arm], greater the chance of 
            simulating a hit. 

        Returns
        -------
        [bool] Hit or miss for the chosen arm

        """
        return self._rand() < self.prob_array[arm]

    def __len__(self):
        return len(self.prob_array)


class BayesianStrategy(object):
    """
    Class to implement Thompson sampling strategy for multi-armed bandit. 
    """

    def __init__(self, bandits, learning_rate=1.):
        """
        Set up for Thompson sampling 

        Parameters
        ----------
        bandits : Bandit simulation object
            Object contains knowledge of true (hidden) probabilities and 
            implements a get_reward() method

        learning_rate: float
            measure of how adaptive the learning is. 

            If learning_rate < 1, the algorithm forgets previous results quicker. 

            learning_rate > 1 implies algorithm bets on earlier winners more 
            often and doesn't explore as much. This amounts to more resistance
            to changing environments.

        """
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.learning_rate = learning_rate
        self.wins = np.zeros(n_bandits)  # Initialize wins of all arms to 0
        self.trials = np.zeros(n_bandits)
        self.total_pulls = 0  # Track number of times any arm is pulled
        self.choices_history = []  # Track history

    def sample_bandits(self, n=1):
        """
        Implements Thompson sampling. 

        The Thompson sampling algorithm proceeds as follows:\
        1. Sample a random variable from the prior of bandit b, for all b
        2. Select the bandit B with the largest sample
        3. Observe the result of pulling bandit B (this is where we simulate the 
           results using get_reward() method of Bandit simulation), and update 
           your prior on bandit B.
        4. Go back to step 1

        Parameters
        ----------
        n : int, optional
            Total number of times the simulation is to be run 
            (the default is 1)

        Returns
        -------
        self: updated instance of BayesianStrategy

        """
        choices_history = np.zeros(n)

        for ii in range(n):
            # sample from the bandits's priors, and select the largest sample
            choice = np.argmax(np.random.beta(a=1 + self.wins,
                                              b=1 + self.trials - self.wins))

            # sample from the chosen bandit and simulate a hit/miss
            result = self.bandits.get_reward(arm=choice)

            # update priors and record history
            self.wins[choice] = self.learning_rate*self.wins[choice] + result
            self.trials[choice] = self.learning_rate*self.trials[choice] + 1
            self.total_pulls += 1
            choices_history[ii] = choice

        self.choices_history = np.r_[self.choices_history, choices_history]
        return
