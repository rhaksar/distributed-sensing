from copy import copy, deepcopy
# import matplotlib.pyplot as pyplot
import numpy as np
import os
import pickle
import sys
import time

from filter import merge_beliefs, update_belief, get_image
from scheduling import set_initial_meetings, set_next_meeting, create_joint_plan, create_solo_plan
from uav import UAV
from utilities import Config, rc_to_xy, xy_to_rc

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    settings = Config()
    settings.seed = seed

    # initialize simulator
    dimension = 25
    sim = LatticeForest(settings.dimension, rng=seed)

    node_row = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_col = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_row, node_col = np.meshgrid(node_row, node_col)
    node_locations = np.stack([node_row.T, node_col.T], axis=2)

    # initialize agents
    initial_belief = dict()
    # radius = 7
    # center = (settings.dimension-1)//2
    for key in sim.group.keys():
        element = sim.group[key]
        # exact belief
        initial_belief[key] = np.zeros(len(element.state_space))
        initial_belief[key][element.state] = 1

        # uniform belief
        # initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)

        # uniform belief around center, exact belief else where
        # if np.linalg.norm(np.asarray(key) - np.array([center, center]), ord=np.inf) <= radius:
        #     initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)
        # else:
        #     initial_belief[key] = np.zeros(len(element.state_space))
        #     initial_belief[key][element.state] = 1

        # initial_belief[key] = 0.10*np.ones(len(element.state_space))
        # initial_belief[key][element.state] += 0.70

    team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
            for i in range(settings.team_size)}

    # create schedule
    S = []
    for i in range(1, np.floor(settings.team_size/2).astype(int)+1):
        S.append((2*i-1, 2*i))
    Sprime = []
    for i in range(1, np.floor((settings.team_size-1)/2).astype(int)+1):
        Sprime.append((2*i, 2*i+1))

    # create deployment locations
    corner = xy_to_rc(settings.dimension, np.array([1.5, 1.5])-settings.cell_side_length)
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    for i, s in enumerate(S):
        idx = np.unravel_index(i, (square_size, square_size), order='C')
        position = (corner[0]-idx[0], corner[1]+idx[1])
        for k in s:
            team[k].position = position
            team[k].first = position

    # deploy agents that do not have a meeting in S
    for agent in team.values():
        offset = len(S)+1
        if agent.position is None:
            idx = np.unravel_index(offset, (square_size, square_size), order='C')
            agent.position = (corner[0]-idx[0], corner[1]+idx[1])
            offset += 1

    # for i in range(2, 2*(np.floor(settings.team_size/2).astype(int)-1)+1, 2):
    #     schedule[1].append((i, i+1))
    # schedule[1].append((1, settings.team_size)) if settings.team_size%2==0 \
    #     else schedule[1].append((settings.team_size-1, settings.team_size))
    if settings.team_size%2 == 0:
        for i in range(2, 2*(np.floor(settings.team_size/2).astype(int)), 2):
            schedule[1].append((i, i+1))
    else:
        for i in range(2, 2*(np.floor(settings.team_size/2).astype(int))+1, 2):
            schedule[1].append((i, i+1))

    # for meeting in schedule[1]:
    #     sub_team = [team[i] for i in meeting]
    #     set_next_meeting(sub_team, sim.group, settings)
    # set_initial_meetings(team, schedule, settings)
    set_initial_meetings(team, schedule, cell_locations, settings)
    for meeting in schedule[1]:
        sub_team = [team[i] for i in meeting]
        create_joint_plan(sub_team, sim.group, settings)
    next_meetings = 0

    save_data = dict()
    save_data['schedule'] = schedule
    save_data['settings'] = settings
    save_data['time_series'] = dict()
    save_data['time_series'][0] = {'team': deepcopy(team),
                                   'process_state': sim.dense_state(),
                                   'process_stats': copy(sim.stats)}

    # print('time {0:d}'.format(0))
    # for agent in team.values():
    #     print(agent.label, agent.meetings)
    # print()

    # main loop
    for t in range(1, 201):
        print('time {0:d}'.format(t))
        # deploy agents two at a time at deployment locations
        # [agent.deploy(t, settings) for agent in team.values()]

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
        if (t-1)%settings.meeting_interval==0:

            for meeting in schedule[next_meetings]:
                sub_team = [team[i] for i in meeting]
                [agent.meetings.pop(0) for agent in sub_team]

                merge_beliefs(sub_team)
                set_next_meeting(sub_team, sim.group, cell_locations, settings)
                create_joint_plan(sub_team, sim.group, settings)

            next_meetings = 0 if next_meetings+1 > len(schedule)-1 else next_meetings+1

        # for agent in team.values():
        #     print(agent.label, agent.meetings)
        # print()

        # update agent position
        for agent in team.values():
            # move(agent, sim.group, settings)
            # agent.position = agent.next_position
            # agent.next_position = None
            agent.plan = create_solo_plan(agent, sim.group, settings)
            agent.position = agent.plan[0]
            agent.meetings[0][1] -= 1
        # [move(agent, sim.group, settings) for agent in team.values()]

        # update simulator if necessary
        if t>1 and (t-1)%settings.true_process_update==0:
            sim.update()

        # update agent belief
        for agent in team.values():
            _, observation = get_image(agent, sim, settings)
            advance = False
            if t>1 and (t-1)%settings.estimate_process_update==0:
                advance = True
            agent.belief = update_belief(sim.group, agent.belief, advance, observation, settings, control=None)

        save_data['time_series'][t] = {'team': deepcopy(team),
                                       'process_state': sim.dense_state(),
                                       'process_stats': copy(sim.stats)}

        if sim.end:
            print('process has terminated')
            break

    # filename = 'sim_images/meetings/meetings-' + time.strftime('%d-%b-%Y-%H%M') + '.pkl'
    # with open(filename, 'wb') as handle:
    #     pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
