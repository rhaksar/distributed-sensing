from collections import defaultdict
import numpy as np


def measure_model(element, state, observation, config):
    """
    Measurement model describing the likelihood function p(observation | state).

    :param element: a Tree object from the LatticeForest simulator.
    :param state: state value, from Tree state space, to determine probability.
    :param observation: observation value, from Tree state space, to determine probability.
    :param config: Config class object containing the probability of measuring the correct state,
    p(observation | state) when observation == state.

    :return: the probability value p(observation | state).
    """
    measure_correct = config.measure_correct
    measure_wrong = (1/(len(element.state_space)-1))*(1-measure_correct)

    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


def multiply_probabilities(values, config):
    """
    Helper function to multiply a list of probabilities.

    :param values: an iterable object containing the probabilities to multiply.
    :param config: a Config class object containing the minimum non-zero probability.

    :return: product of probabilities, which is zero if any value is below a specified threshold.
    """

    if any([v < config.threshold for v in values]):
        return 0
    else:
        sum_log = sum([np.log(v) for v in values])
        if sum_log <= np.log(config.threshold):
            return 0
        else:
            return np.exp(sum_log)


def get_image(uav, simulation, config, uncertainty=True):
    """
    Helper function to return an image of the LatticeForest for a UAV, with or without uncertainty.

    :param uav: a UAV object, where the UAV location in (row, col) units is used as the image center.
    :param simulation: a LatticeForest simulation object, whose underlying state is used to generate the image.
    :param config: Config class object containing the image size and measurement model parameters.
    :param uncertainty: a boolean indicating whether or not the true underlying state should be returned.

    :return: a tuple of a 2D numpy array and a dictionary. the numpy array, image, contains the Tree state values
    describing the image. the dictionary consists of {Tree position: observation} key value pairs if uncertainty = True,
    otherwise the dictionary is empty.
    """
    state = simulation.dense_state()
    r0, c0 = uav.position
    image = np.zeros(config.image_size).astype(np.int8)
    observation = dict()

    # center image at UAV position
    half_row = (config.image_size[0]-1)//2
    half_col = (config.image_size[1]-1)//2
    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = r0 + dr
            c = c0 + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                if not uncertainty:
                    image[ri, ci] = state[r, c]
                # sample according to the measurement model to generate an uncertain image
                else:
                    element = simulation.group[(r, c)]
                    probs = [measure_model(element, element.state, o, config) for o in element.state_space]
                    obs = np.random.choice(element.state_space, p=probs)
                    observation[(r, c)] = obs
                    image[ri, ci] = obs

    return image, observation


def merge_beliefs(sub_team):
    """
    Helper function to merge the beliefs of multiple UAVs by averaging.

    :param sub_team: a list of UAV class objects.

    :return: the merged belief, described by a dictionary of
    {Tree position: list of probabilities for each state value} for the LatticeForest.
    """
    merged = dict()
    for key in sub_team[0].belief.keys():
        beliefs = [agent.belief[key] for agent in sub_team]
        merged[key] = sum(beliefs)/len(beliefs)
        merged[key] /= np.sum(merged[key])

    return merged


def update_belief(simulation_group, prior, advance, observation, config, control=None):
    """
    Update the belief given an observation for the LatticeForest simulator using an approximate filtering scheme.

    :param simulation_group: dictionary containing {Tree position: Tree object} key value pairs for the LatticeForest
    simulator. this is used to iterate through each element and update the corresponding belief.
    :param prior: a dictionary describing the prior belief, as {Tree position: list of probabilities for
    each state value}.
    :param advance: boolean indicating whether or not the LatticeForest is updated at the current time step.
    :param observation: dictionary describing the belief, as {Tree position: tree_observation} key value pairs.
    the value tree_observation may be an integer (single observation) or a list of integers (several observations of
    the same tree at a single time step).
    :param config: a Config class object containing the measurement model parameters.
    :param control: a dictionary describing the control effort applied at the current time step, as
    {Tree position: (delta_alpha, delta_beta)} key value pairs. this parameter defaults to None, indicating no control
    effort.

    :return: the updated belief as a dictionary of {Tree position: list of probabilities for each time step} key value
    pairs.
    """
    if control is None:
        control = defaultdict(lambda: (0, 0))

    posterior = dict()
    for key in simulation_group.keys():
        element = simulation_group[key]
        element_posterior = np.zeros(len(element.state_space))
        num_neighbors = len(element.neighbors)
        on_fire = 1

        # the Tree dynamics are based on the number of neighbors on fire rather than the identity of the neighboring
        # Trees. as a result, iterate over the possible number of Trees on fire to consider different state transitions.
        caf = np.zeros(num_neighbors+1)
        for state_idx in range(2**num_neighbors):
            xj = np.base_repr(state_idx, base=2).zfill(num_neighbors)
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

            caf[active] += multiply_probabilities(values, config)

        # perform open-loop dynamics update
        for x_t in element.state_space:
            for x_tm1 in element.state_space:
                # if the simulator is updated, then consider possible state transitions
                if advance:
                    for active in range(num_neighbors+1):
                        values = [element.dynamics((x_tm1, active, x_t), control[key]), caf[active], prior[key][x_tm1]]
                        element_posterior[x_t] += multiply_probabilities(values, config)
                # otherwise, the dynamics are static
                else:
                    element_posterior[x_t] += (x_t == x_tm1)*prior[key][x_tm1]

        # adjust dynamics update based on observation(s)
        for x_t in element.state_space:
            if key in observation.keys():
                # check for single observation or multiple observations
                if type(observation[key]) == int:
                    element_posterior[x_t] *= measure_model(element, x_t, observation[key], config)
                elif type(observation[key]) == list:
                    element_posterior[x_t] *= np.prod([measure_model(element, x_t, obs, config)
                                                       for obs in observation[key]])

        posterior[key] = element_posterior/np.sum(element_posterior)

    return posterior
