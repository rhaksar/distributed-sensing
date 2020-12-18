from framework.filter import get_image
import numpy as np


def compute_accuracy(belief, true_state, config):
    accuracy = 0
    for key in belief.keys():
        if np.argmax(belief[key]) == true_state[key[0], key[1]]:
            accuracy += 1

    return accuracy/(config.dimension**2)


def compute_frequency(team, true_state, data):
    unique = list()
    for agent in team.values():
        if agent.position not in unique and true_state[agent.position[0], agent.position[1]] == 1:
            data[agent.position[0], agent.position[1]] += 1
            unique.append(data)


def compute_coverage(team, sim_object, state, settings):
    if len(sim_object.fires) == 0:
        return 0

    team_observation = set()
    for agent in team.values():
        _, observation = get_image(agent, sim_object, settings)
        agent_observation = {key for key in observation.keys() if state[key[0], key[1]] == 1}
        team_observation |= agent_observation

    return len(team_observation)/len(sim_object.fires)
