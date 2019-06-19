import copy
import numpy as np
import time


def multiply_probabilities(values):
    threshold = 1e-20  # determines the smallest non-zero probability
    if any([v < threshold for v in values]):
        return 0
    else:
        sum_log = sum([np.log(v) for v in values])
        if sum_log <= np.log(threshold):
            return 0
        else:
            return np.exp(sum_log)


class ApproxFilter(object):

    def __init__(self, belief):
        self.belief = belief

    def filter(self, simulation, element_update):
        tic = time.clock()
        posterior = copy.copy(self.belief)

        for key in simulation.group.keys():
            element = simulation.group[key]
            posterior[key] = element_update(key, element, self.belief)

        toc = time.clock()
        self.belief = posterior
        return posterior, toc-tic
