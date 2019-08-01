from copy import copy
import numpy as np
import os
import sys

from filter import merge_beliefs, update_belief, get_image
from scheduling import set_initial_meetings, set_next_meeting, create_joint_plan, move
from uav import UAV
from utilities import Config, xy_to_rc

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
        initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)  # uniform belief

    team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
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

    schedule = [[], []]
    settings.total_interval = 2*settings.meeting_interval
    for i in range(1, 2*np.floor(settings.team_size/2).astype(int)-1+1, 2):
        schedule[0].append((i, i+1))

    corner = xy_to_rc(settings.dimension, np.array([1.5, 1.5])-settings.cell_side_length)
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    for i in range(len(schedule[0])):
        idx = np.unravel_index(i, (square_size, square_size), order='C')
        # position = corner + 0.5*np.asarray(idx[::-1])
        position = (corner[0]+idx[0], corner[1]+idx[1])
        for idx in schedule[0][i]:
            team[idx].position = position
            team[idx].meetings.append([position, settings.meeting_interval])

    for agent in team.values():
        offset = len(schedule[0])+1
        if agent.position is None:
            idx = np.unravel_index(offset, (square_size, square_size), order='C')
            # agent.position = corner + 0.5*np.asarray(idx[::-1])
            agent.position = (corner[0]+idx[0], corner[1]+idx[1])
            offset += 1

    for i in range(2, 2*(np.floor(settings.team_size/2).astype(int)-1)+1, 2):
        schedule[1].append((i, i+1))
    schedule[1].append((1, settings.team_size)) if settings.team_size%2==0 \
        else schedule[1].append((settings.team_size-1, settings.team_size))

    # for meeting in schedule[1]:
    #     sub_team = [team[i] for i in meeting]
    #     set_next_meeting(sub_team, sim.group, settings)
    # set_initial_meetings(team, schedule, settings)
    set_initial_meetings(team, schedule, sim.group, settings)
    next_meetings = 0

    # main loop
    for t in range(1, 2):
        # deploy agents two at a time at deployment locations
        # [agent.deploy(t, settings) for agent in team.values()]

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
        if (t-1)%settings.meeting_interval==0:

            for meeting in schedule[next_meetings]:
                sub_team = [team[i] for i in meeting]
                [agent.meetings.pop(0) for agent in sub_team]

                merge_beliefs(sub_team)
                set_next_meeting(sub_team, sim.group, settings)
                create_joint_plan(sub_team, sim.group, settings)

            next_meetings = 0 if next_meetings >= len(schedule) else next_meetings+1

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

        # update agent position
        for agent in team.values():
            move(agent, sim.group, settings)
            agent.position = agent.next_position
            agent.next_position = None
        # [move(agent, sim.group, settings) for agent in team.values()]

        # update simulator if necessary
        if t%settings.true_process_update==0:
            sim.update()

        # update agent belief
        for agent in team.values():
            _, observation = get_image(agent, sim, settings)
            advance = False
            if t%settings.estimate_process_update==0:
                advance = True
            agent.belief = update_belief(sim.group, agent.belief, advance, observation, control=None)

        # plot image - left is ground truth, right is agent paths and meeting locations

    print('done')
