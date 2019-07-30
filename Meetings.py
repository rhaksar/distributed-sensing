from copy import copy
import numpy as np
import os
import sys

from uav import UAV
from utilities import Config

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    settings = Config()

    # initialize simulator
    dimension = 25
    sim = LatticeForest(dimension, rng=seed)

    # initialize agents
    initial_belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)

    team = {i+1: UAV(id=i+1, belief=copy(initial_belief), image_size=settings.image_size)
            for i in range(settings.team_size)}

    # specify graph schedule
    for i in range(1, 2*int(np.floor(settings.team_size/2))-1+1, 2):
        team[i].meetings.append((i+1, None))
        team[i+1].meetings.append((i, None))

    for i in range(2, 2*(int(np.floor(settings.team_size/2))-1)+1, 2):
        team[i].meetings.append((i+1, None))
        team[i+1].meetings.append((i, None))
    team[1].meetings.append((settings.team_size, None))
    team[settings.team_size].meetings.append((1, None))

    # main loop
    for t in range(1, 3):
        # deploy agents two at a time at deployment locations
        [agent.deploy(t, settings) for agent in team.values()]

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths

        # possible optimization for setting the meeting location:
        #   run filter forward for T steps, where T is the next meeting time, and incorporate expected conditional
        #   entropy reduction due to traveling to other meetings
        #   choose the location with the highest expected entropy which is within T steps away from all agents (check 
        #   distances with Linf norm)

        # for agents not in a meeting, individually plan a path

        # for each agent: update position using plan and update belief using image

        # plot image - left is ground truth, right is agent paths and meeting locations

    print('done')
