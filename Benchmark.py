from copy import copy
import numpy as np
import os
import pickle
import sys
import time

from filter import merge_beliefs, update_belief, get_image
from metrics import compute_coverage
from scheduling import schedule_initial_meetings, schedule_next_meeting, \
    create_joint_plan, create_solo_plan
from uav import UAV
from utilities import Config

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest


if __name__ == '__main__':
    print('[Benchmark] started at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()

    total_simulations = 10
    offset = 10
    rho = 1
    total_iterations = rho*60 + 1

    if len(sys.argv) != 3:
        tau = 4
        C = 2
    else:
        tau = int(sys.argv[1])
        C = int(sys.argv[2])

    pc = 0.95
    print('[Benchmark] rho = ' + str(rho) + ', tau = ' + str(tau) + ', C = ' + str(C) + ', pc = ' + str(pc))

    settings = Config(process_update=rho, team_size=C, meeting_interval=tau, measure_correct=pc)
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)

    # initialize simulator
    sim = LatticeForest(settings.dimension)

    node_row = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_col = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_row, node_col = np.meshgrid(node_row, node_col)
    node_locations = np.stack([node_row.T, node_col.T], axis=2)

    # create schedule
    S = []
    for i in range(1, np.floor(settings.team_size/2).astype(int)+1):
        S.append((2*i-1, 2*i))
    Sprime = []
    for i in range(1, np.floor((settings.team_size-1)/2).astype(int)+1):
        Sprime.append((2*i, 2*i+1))

    # initialize agents
    initial_belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        # exact belief
        initial_belief[key] = np.zeros(len(element.state_space))
        initial_belief[key][element.state] = 1

        # exact for healthy, uniform otherwise
        # if element.state == 0:
        #     initial_belief[key] = np.array([1.0, 0.0, 0.0])
        # else:
        #     initial_belief[key] = (1/3)*np.ones(len(element.state_space))

        # uniform uncertainty
        # initial_belief[key] = np.ones(len(element.state_space))/len(element.state_space)

    # initialize data structure for saving information
    save_data = dict()

    for sim_count, seed in enumerate(range(offset, total_simulations+offset)):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()
        save_data[seed] = dict()

        team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
                for i in range(settings.team_size)}

        # deploy agents
        for ith_meeting, s in enumerate(S):
            idx = np.unravel_index(ith_meeting, (square_size, square_size), order='C')
            position = (settings.corner[0] - idx[0], settings.corner[1] + idx[1])
            for k in s:
                team[k].position = position
                team[k].first = position

        # deploy remaining agents that do not have a meeting in S
        offset = len(S)+1
        for agent in team.values():
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

        next_meetings = 0

        # frequency = np.zeros((settings.dimension, settings.dimension))
        # state = sim.dense_state()
        # save_data[seed][0] = [compute_accuracy(team[label].belief, state, settings) for label in team.keys()]
        save_data[seed]['coverage'] = []

        # main loop
        for t in range(1, total_iterations):
            # check if any meetings should occur
            #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
            if (t-1) % settings.meeting_interval == 0:

                for s in [S, Sprime][next_meetings]:
                    sub_team = [team[k] for k in s]

                    merged_belief = merge_beliefs(sub_team)
                    for agent in sub_team:
                        agent.belief = copy(merged_belief)

                    meeting = schedule_next_meeting(sub_team, merged_belief, sim.group, node_locations, settings)
                    # print('meeting', s, 'chose location', meeting)
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
                agent.plan = create_solo_plan(agent, sim.group, settings)
                # print('agent {0:d}: {1} -> {2} -> ... -> {3}'.format(agent.label, agent.position,
                #                                                      agent.plan[0], agent.first))

                agent.position = agent.plan[0]
                for other_label in agent.other_plans.keys():
                    if not agent.other_plans[other_label]:
                        continue
                    agent.other_plans[other_label].pop(0)
                agent.budget -= 1

            # update agent belief
            for agent in team.values():
                _, observation = get_image(agent, sim, settings)
                advance = False
                if t > 1 and (t-1) % settings.process_update == 0:
                    advance = True
                agent.belief = update_belief(sim.group, agent.belief, advance, observation, settings, control=None)

            # update simulator if necessary
            if t > 1 and (t-1) % settings.process_update == 0:
                sim.update()

            state = sim.dense_state()
            current_coverage = compute_coverage(team, sim, state, settings)
            # print('time {0:d} coverage = {1:0.4f}'.format(t, current_coverage))
            save_data[seed]['coverage'].append(current_coverage)
            # compute_frequency(team, state, frequency)
            # save_data[seed][t] = [compute_accuracy(team[label].belief, state, settings) for label in team.keys()]

        # save_data[seed][total_iterations] = frequency
        print('[Benchmark] finished simulation {0:d} (coverage = {1:0.4f})'.format(sim_count+1,
                                                                                   np.mean(save_data[seed]['coverage'])))

    # write data to file
    filename = 'Benchmark/benchmark-' + 'rho' + str(rho).zfill(2) + \
               'tau' + str(tau).zfill(2) + 'C' + str(C).zfill(2) + 'pc' + str(pc) + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('[Benchmark] mean coverage = {0:0.4f}'.format(np.mean([np.mean(save_data[seed]['coverage'])
                                                                 for seed in save_data.keys()])))

    toc = time.clock()
    dt = toc - tic
    print('[Benchmark] completed at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    print('[Benchmark] %0.2fs = %0.2fm = %0.2fh elapsed' % (dt, dt / 60, dt / 3600))
