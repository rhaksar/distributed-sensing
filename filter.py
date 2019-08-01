from collections import defaultdict
from copy import copy
import numpy as np
import time


def measure_model(element, state, observation):
    measure_correct = 0.8
    measure_wrong = (1/(len(element.state_space)-1))*(1-measure_correct)

    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


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


def get_image(uav, simulation, config, uncertainty=True):
    state = simulation.dense_state()
    r0, c0 = uav.position
    image = np.zeros(config.image_size).astype(np.int8)
    observation = dict()

    half_row = (config.image_size[0]-1)//2
    half_col = (config.image_size[1] - 1)//2
    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = r0 + dr
            c = c0 + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                if not uncertainty:
                    image[ri, ci] = state[r, c]
                else:
                    element = simulation.group[(r, c)]
                    probs = [measure_model(element, element.state, o) for o in element.state_space]
                    obs = np.random.choice(element.state_space, p=probs)
                    observation[(r, c)] = obs
                    image[ri, ci] = obs

    return image, observation


def merge_beliefs(sub_team):
    merged = dict()
    for key in sub_team[0].belief.keys():
        beliefs = [agent.belief[key] for agent in sub_team]
        merged[key] = sum(beliefs)/len(beliefs)

    for agent in sub_team:
        agent.belief = copy(merged)
    return


def update_belief(simulation_group, prior, advance, observation, control=None):
    if control is None:
        control = defaultdict(lambda: (0, 0))

    posterior = copy(prior)
    for key in simulation_group.keys():
        element = simulation_group[key]
        element_posterior = np.zeros(len(element.state_space))
        num_neighbors = len(element.neighbors)
        on_fire = 1

        caf = np.zeros(num_neighbors+1)
        for l in range(2**num_neighbors):
            xj = np.base_repr(l, base=2).zfill(num_neighbors)
            active = xj.count('1')

            values = []
            for n in range(num_neighbors):
                neighbor_key = element.neighbors[n]
                prob = None
                if int(xj[n]) == 0:
                    prob = 1 - prior[neighbor_key][on_fire]
                elif int(xj[n]) == 1:
                    prob = prior[neighbor_key][on_fire]

                values.append(prob)

            caf[active] += multiply_probabilities(values)

        for x_t in element.state_space:
            for x_tm1 in element.state_space:
                if advance:
                    for active in range(num_neighbors+1):
                        values = [element.dynamics((x_tm1, active, x_t), control[key]), caf[active], prior[key][x_tm1]]
                        element_posterior[x_t] += multiply_probabilities(values)
                else:
                    element_posterior[x_t] += (x_t == x_tm1)*prior[key][x_tm1]

        for x_t in element.state_space:
            if key in observation.keys():
                element_posterior[x_t] *= measure_model(element, x_t, observation[key])
            else:
                weight = 0.01
                mean = np.mean(element_posterior)
                element_posterior = [v - weight*np.sign(v-mean)*np.abs(v-mean) for v in element_posterior]

        posterior[key] = element_posterior/np.sum(element_posterior)

    return posterior


class ApproxFilter(object):

    def __init__(self, belief):
        self.belief = belief

    def filter(self, simulation, element_update):
        tic = time.clock()
        posterior = copy(self.belief)

        for key in simulation.group.keys():
            element = simulation.group[key]
            posterior[key] = element_update(key, element, self.belief)

        toc = time.clock()
        self.belief = posterior
        return posterior, toc-tic
