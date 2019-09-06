from copy import copy, deepcopy
import numpy as np
import os
import pickle
import sys
import time

from filter import merge_beliefs, update_belief, get_image
from scheduling import schedule_initial_meetings, schedule_next_meeting, \
    create_joint_plan, create_solo_plan, compute_entropy
from uav import UAV
from utilities import Config

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest


def compute_accuracy(belief, true_state, config):
    accuracy = 0
    for key in belief.keys():
        if np.argmax(belief[key]) == true_state[key[0], key[1]]:
            accuracy += 1

    return accuracy/(config.dimension**2)


if __name__ == '__main__':
    print('[Meetings] started at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()

    total_iterations = 121
    seed = 2
    np.random.seed(seed)
    settings = Config(team_size=10, meeting_interval=4, measure_correct=0.95)
    settings.seed = seed

    # initialize simulator
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

    team = {i+1: UAV(label=i+1, belief=initial_belief, image_size=settings.image_size)
            for i in range(settings.team_size)}

    # create schedule
    S = []
    for i in range(1, np.floor(settings.team_size/2).astype(int)+1):
        S.append((2*i-1, 2*i))
    Sprime = []
    for i in range(1, np.floor((settings.team_size-1)/2).astype(int)+1):
        Sprime.append((2*i, 2*i+1))

    # deploy agents
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    for ith_meeting, s in enumerate(S):
        idx = np.unravel_index(ith_meeting, (square_size, square_size), order='C')
        position = (settings.corner[0]-idx[0], settings.corner[1]+idx[1])
        for k in s:
            team[k].position = position
            team[k].first = position

    # deploy remaining agents that do not have a meeting in S
    for agent in team.values():
        offset = len(S)+1
        if agent.position is None:
            idx = np.unravel_index(offset, (square_size, square_size), order='C')
            agent.position = (settings.corner[0]-idx[0], settings.corner[1]+idx[1])
            offset += 1

    # set initial meeting locations and time budgets
    meetings = schedule_initial_meetings(team, Sprime, sim.group, node_locations, settings)
    for label in meetings.keys():
        team[label].last = meetings[label]
        team[label].budget = settings.meeting_interval

    # make sure all agents have valid first and last meeting locations, as well as correct time budgets
    for agent in team.values():
        if agent.first is None:
            agent.first = agent.last
            agent.budget = settings.meeting_interval
        if agent.last is None:
            agent.last = agent.first
            agent.budget = 2*settings.meeting_interval

    # should this still be done here?
    # for s in Sprime:
    #     sub_team = [team[k] for k in s]
    #     create_joint_plan(sub_team, sim.group, settings)

    next_meetings = 0

    save_data = dict()
    save_data['schedule'] = [S, Sprime]
    save_data['settings'] = settings
    save_data['time_series'] = dict()
    state = sim.dense_state()
    save_data['time_series'][0] = {'entropy': {label: compute_entropy(team[label].belief, settings) for label in
                                               team.keys()},
                                   'accuracy': {label: compute_accuracy(team[label].belief, state, settings)
                                                for label in team.keys()},
                                   'position': {label: copy(team[label].position) for label in team.keys()},
                                   'plan': {label: copy(team[label].plan) for label in team.keys()},
                                   'process_state': state}
    # save_data['time_series'][0] = {'team': deepcopy(team),
    #                                'process_state': state}

    # main loop
    for t in range(1, total_iterations):
        print('[Meetings] time {0:d}'.format(t))
        # deploy agents two at a time at deployment locations
        # [agent.deploy(t, settings) for agent in team.values()]

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
        if (t-1) % settings.meeting_interval == 0:

            for s in [S, Sprime][next_meetings]:
                # print('meeting', s)
                sub_team = [team[k] for k in s]
                assert sub_team[0].position == sub_team[1].position

                merged_belief = merge_beliefs(sub_team)
                for agent in sub_team:
                    agent.belief = merged_belief

                meeting = schedule_next_meeting(sub_team, merged_belief, sim.group, node_locations, settings)
                # print('chose location', meeting)
                for agent in sub_team:
                    # if agent.first == agent.last:
                    if agent.label in [1, settings.team_size]:
                        agent.first = meeting
                        agent.last = meeting
                        agent.budget = 2*settings.meeting_interval
                    else:
                        agent.first = agent.last
                        agent.last = meeting
                        agent.budget = settings.meeting_interval

                plans = create_joint_plan(sub_team, sim.group, settings)
                for agent in sub_team:
                    for label in plans.keys():
                        if label == agent.label:
                            continue
                        agent.other_plans[label] = copy(plans[label])

            next_meetings = 0 if next_meetings+1 > 1 else next_meetings+1

        # update agent position

        for agent in team.values():
            # print(agent.label, agent.position)
            agent.plan = create_solo_plan(agent, sim.group, settings)
            agent.position = agent.plan[0]
            for other_label in agent.other_plans.keys():
                if not agent.other_plans[other_label]:
                    continue
                agent.other_plans[other_label].pop(0)
            agent.budget -= 1

        # update simulator if necessary
        if t > 1 and (t-1) % settings.process_update == 0:
            sim.update()

        # update agent belief
        for agent in team.values():
            _, observation = get_image(agent, sim, settings)
            advance = False
            if t > 1 and (t-1) % settings.process_update == 0:
                advance = True
            agent.belief = update_belief(sim.group, agent.belief, advance, observation, settings, control=None)

        # save_data['time_series'][t] = {'team': deepcopy(team),
        #                                'process_state': sim.dense_state()}
        state = sim.dense_state()
        save_data['time_series'][t] = {'entropy': {label: compute_entropy(team[label].belief, settings) for label in
                                                   team.keys()},
                                       'accuracy': {label: compute_accuracy(team[label].belief, state, settings)
                                                    for label in team.keys()},
                                       'position': {label: copy(team[label].position) for label in team.keys()},
                                       'plan': {label: copy(team[label].plan) for label in team.keys()},
                                       'process_state': state}
        # save_data['time_series'][t] = {'team': deepcopy(team),
        #                                'process_state': state}

        # if sim.early_end:
        #     print('process cannot spread')
        #     break
        #
        # if sim.end:
        #     print('process has terminated')
        #     break

    # write data to file
    filename = 'sim_images/meetings/meetings-' + time.strftime('%d-%b-%Y-%H%M') + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    toc = time.clock()
    dt = toc - tic
    print('[Meetings] completed at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    print('[Meetings] %0.2fs = %0.2fm = %0.2fh elapsed' % (dt, dt / 60, dt / 3600))
