from copy import copy
import numpy as np
import pickle
import time

from framework.filter import merge_beliefs, update_belief, get_image
from framework.metrics import compute_coverage
from framework.scheduling import schedule_initial_meetings, schedule_next_meeting, \
    create_joint_plan, create_solo_plan, compute_entropy
from framework.uav import UAV
from framework.utilities import Config

from simulators.fires.LatticeForest import LatticeForest


if __name__ == '__main__':
    print('[Meetings] started at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()

    total_iterations = 61
    seed = 0 + 10
    np.random.seed(seed)
    settings = Config(process_update=1, team_size=5, meeting_interval=8, measure_correct=0.95)
    settings.seed = seed

    # initialize simulator
    sim = LatticeForest(settings.dimension, rng=seed)

    # optional - add fires to initial condition
    # initial_fire = sim.fires
    # initial_fire.extend([(6, 6), (6, 5), (6, 7), (7, 6), (5, 6), (5, 7), (5, 8), (4, 7)])
    # sim.initial_fire = initial_fire
    # sim.reset()

    # initialize agents
    initial_belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        # exact belief
        initial_belief[key] = np.zeros(len(element.state_space))
        initial_belief[key][element.state] = 1

    team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
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
    offset = len(S)+1
    for agent in team.values():
        if not agent.position:
            idx = np.unravel_index(offset, (square_size, square_size), order='C')
            agent.position = (settings.corner[0]-idx[0], settings.corner[1]+idx[1])
            offset += 1

    # set initial meeting locations and time budgets
    meetings = schedule_initial_meetings(team, Sprime, sim.group, settings)
    for label in meetings.keys():
        team[label].last = meetings[label]
        team[label].budget = settings.meeting_interval

    # make sure all agents have valid first and last meeting locations, as well as correct time budgets
    for agent in team.values():
        if not agent.first:
            agent.first = agent.last
            agent.budget = settings.meeting_interval
        if not agent.last:
            agent.last = agent.first
            agent.budget = 2*settings.meeting_interval

    next_meetings = 0

    # date to write to file
    save_data = dict()
    save_data['schedule'] = [S, Sprime]
    save_data['settings'] = settings
    save_data['time_series'] = dict()
    state = sim.dense_state()
    save_data['time_series'][0] = {'entropy': {label: compute_entropy(team[label].belief, settings) for label in
                                               team.keys()},
                                   'position': {label: copy(team[label].position) for label in team.keys()},
                                   'plan': {label: copy(team[label].plan) for label in team.keys()},
                                   'process_state': state}
    coverage_metric_series = []

    # main loop
    for t in range(1, total_iterations):
        print('[Meetings] time {0:d}'.format(t))

        # check if any meetings should occur
        #   agents in a meeting merge beliefs, set next meeting based on schedule+filter, and jointly plan paths
        if (t-1) % settings.meeting_interval == 0:

            for s in [S, Sprime][next_meetings]:
                sub_team = [team[k] for k in s]
                assert sub_team[0].position == sub_team[1].position

                merged_belief = merge_beliefs(sub_team)
                for agent in sub_team:
                    agent.belief = copy(merged_belief)

                meeting = schedule_next_meeting(sub_team, merged_belief, sim.group, settings)
                for agent in sub_team:
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

        # save data
        coverage_metric_series.append(compute_coverage(team, sim, settings))
        save_data['time_series'][t] = {'entropy': {label: compute_entropy(team[label].belief, settings) for label in
                                                   team.keys()},
                                       'position': {label: copy(team[label].position) for label in team.keys()},
                                       'plan': {label: copy(team[label].plan) for label in team.keys()},
                                       'process_state': state}

    # write data to file
    filename = 'Meetings/meetings-' + time.strftime('%d-%b-%Y-%H%M') + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_data['coverage'] = coverage_metric_series
    # print('coverage metric =',save_data['coverage'])

    toc = time.clock()
    dt = toc - tic
    print('[Meetings] completed at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    print('[Meetings] %0.2fs = %0.2fm = %0.2fh elapsed' % (dt, dt/60, dt/3600))
