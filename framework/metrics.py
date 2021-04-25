from framework.filter import get_image
import numpy as np


def compute_accuracy(belief, true_state, config):
    """
    The accuracy metric computes the maximum likelihood state from a belief and compares it to the ground truth, for a
    single time step.

    :param belief: dictionary representing the belief,
    consisting of {Tree position: list of probabilities for each state} key value pairs.
    :param true_state: 2D numpy array where each (row, col) value corresponds to the true state of the Tree at the
    position (row, col).
    :param config: a Config class object containing the size of the LatticeForest.

    :return: the accuracy metric as a percentage.
    """
    accuracy = 0
    for key in belief.keys():
        if np.argmax(belief[key]) == true_state[key[0], key[1]]:
            accuracy += 1

    return accuracy/(config.dimension**2)


def compute_frequency(team, true_state):
    """
    The frequency metric returns the unique UAV positions which correspond to a tree on fire based on the true state.

    :param team: a list of UAV class objects representing the team.
    :param true_state: a 2D numpy array representing the true state of the LatticeForest.

    :return: a 2D numpy array, with a value of 1 where a tree on fire corresponds to an agent position. the array is
    zero otherwise.
    """
    data = np.zeros_like(true_state)

    unique = list()
    for agent in team.values():
        if agent.position not in unique and true_state[agent.position[0], agent.position[1]] == 1:
            data[agent.position[0], agent.position[1]] += 1
            unique.append(agent.position)

    return data


def compute_coverage(team, sim_object, config):
    """
    The coverage metric computes the number of unique trees on fires observed by a team of UAVs, based on the true
    state, divided by the current number of trees on fire.

    :param team: a list of UAV class objects representing the team.
    :param sim_object: a LatticeForest simulator object.
    :param config: a Config class object containing the measurement model parameters.

    :return: the coverage metric value at the current time step.
    """
    state = sim_object.dense_state()
    if len(sim_object.fires) == 0:
        return 0

    team_observation = set()
    for agent in team.values():
        _, observation = get_image(agent, sim_object, config)
        agent_observation = {key for key in observation.keys() if state[key[0], key[1]] == 1}
        team_observation |= agent_observation

    return len(team_observation)/len(sim_object.fires)
