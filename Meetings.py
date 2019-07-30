from copy import copy
import numpy as np
import os
import sys

from filter import merge_beliefs
from scheduling import set_initial_meetings, set_next_meeting
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
    sim = LatticeForest(settings.dimension, rng=seed)

    # initialize agents
    initial_belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)

    team = {i+1: UAV(id=i+1, belief=copy(initial_belief), image_size=settings.image_size)
            for i in range(settings.team_size)}

    # specify graph schedule
    # for i in range(1, 2*int(np.floor(settings.team_size/2))-1+1, 2):
    #     team[i].meetings.append((oi+1, None))
    #     team[i+1].meetings.append((i, None))
    #
    # for i in range(2, 2*(int(np.floor(settings.team_size/2))-1)+1, 2):
    #     team[i].meetings.append((i+1, None))
    #     team[i+1].meetings.append((i, None))
    # team[1].meetings.append((settings.team_size, None))
    # team[settings.team_size].meetings.append((1, None))

    schedule = [[]]*2
    settings.total_interval = 2*settings.meeting_interval
    for i in range(1, 2*np.floor(settings.team_size/2).astype(int)-1+1, 2):
        schedule[0].append((i, i+1))

    corner = np.array([1.5, 1.5])
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    for i in range(len(schedule[0])):
        idx = np.unravel_index(i, (square_size, square_size), order='C')
        position = corner + 0.5*np.asarray(idx[::-1])
        for idx in schedule[0][i]:
            team[idx].position = position

    for i in range(2, 2*(np.floor(settings.team_size/2).astype(int)-1)+1, 2):
        schedule[1].append((i, i+1))
    schedule[1].append((1, settings.team_size)) if settings.team_size%2==0 \
        else schedule[1].append((settings.team_size-1, settings.team_size))
    set_initial_meetings(team, schedule, settings)
    next_set = 1

    # main loop
    for t in range(1, 2):
        # deploy agents two at a time at deployment locations
        # [agent.deploy(t, settings) for agent in team.values()]

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
        if (t-1)%settings.meeting_interval==0:

            for meeting in schedule[next_set]:
                sub_team = [team[i] for i in meeting]
                merge_beliefs(sub_team)

                set_next_meeting(sub_team)

            next_set = 0 if next_set>=len(schedule) else next_set+1

        # possible optimization for setting the meeting location:
        #   run filter forward for T steps, where T is the next meeting time, and incorporate expected conditional
        #   entropy reduction due to traveling to other meetings
        #   choose the location with the highest expected entropy which is within T steps away from all agents (check 
        #   distances with Linf norm)

        # possible optimization for setting the meeting location:
        #   run filter forward for T steps on the merged belief, where T is the next meeting time, and incorporate
        #   expected conditional entropy reduction due to traveling to other meetings
        #   find the longest path of length T and use the final location as the next meeting location

        # for agents not in a meeting, individually plan a path

        # for each agent: update position using plan and update belief using image

        # plot image - left is ground truth, right is agent paths and meeting locations

    print('done')
