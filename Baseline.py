from copy import copy
import numpy as np
import os
from operator import itemgetter
import pickle
import scipy.ndimage as sn
import sys
import time

from filter import merge_beliefs, update_belief, get_image
from metrics import compute_coverage
from scheduling import schedule_initial_meetings, create_solo_plan, \
    compute_conditional_entropy, graph_search, update_information
from uav import UAV
from utilities import Config

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest


if __name__ == '__main__':
    print('[Baseline] started at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()

    communication = False
    if communication:
        print('[Baseline]     with communication')

    total_simulations = 10
    offset = 10
    rho = 1
    total_iterations = 61
    tau = 8
    C = 2
    pc = 0.95
    print('[Baseline] tau = ' + str(tau) + ', C = ' + str(C) + ', pc = ' + str(pc))

    settings = Config(process_update=rho, team_size=C, meeting_interval=tau, measure_correct=pc)
    # square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    square_size = np.ceil(np.sqrt(settings.team_size)).astype(int)

    # initialize simulator
    sim = LatticeForest(settings.dimension)

    node_row = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_col = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_row, node_col = np.meshgrid(node_row, node_col)
    node_locations = np.stack([node_row.T, node_col.T], axis=2)

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
        sim.seed = seed
        sim.reset()
        save_data[seed] = dict()

        team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
                for i in range(settings.team_size)}

        if communication:
            team_belief = copy(initial_belief)

        # deploy all agents at unique locations
        offset = 0
        for agent in team.values():
            agent.budget = settings.meeting_interval
            if agent.position is None:
                idx = np.unravel_index(offset, (square_size, square_size), order='C')
                agent.position = (settings.corner[0]-idx[0], settings.corner[1]+idx[1])
                offset += 1

        # frequency = np.zeros((settings.dimension, settings.dimension))
        # state = sim.dense_state()
        # save_data[seed][0] = [compute_accuracy(team[label].belief, state, settings) for label in team.keys()]
        save_data[seed]['coverage'] = []

        # main loop
        for t in range(1, total_iterations):

            if communication:
                # merge all agent beliefs
                # merged_belief = merge_beliefs([agent for agent in team.values()])
                # for agent in team.values():
                #     agent.belief = merged_belief

                # predict future belief of team belief (open-loop)
                predicted_belief = copy(team_belief)
                belief_updates = settings.meeting_interval//settings.process_update
                for _ in range(belief_updates):
                    predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

                # find locations of high entropy and use them as planned locations
                conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
                conditional_entropy += 0.1
                meetings = dict()

                for agent in team.values():

                    weights = sn.filters.convolve(conditional_entropy,
                                                  np.ones(settings.image_size),
                                                  mode='constant', cval=0)

                    distances = np.linalg.norm(node_locations - agent.position, ord=np.inf, axis=2)
                    locations_r, locations_c = np.where(distances == settings.meeting_interval)
                    locations = list(zip(locations_r, locations_c))

                    if len(locations) == 1:
                        chosen_location = locations[0]
                    else:
                        np.random.shuffle(locations)
                        options = [(weights[r, c], (r, c)) for (r, c) in locations]
                        chosen_location = max(options, key=itemgetter(0))[1]

                    meetings[agent.label] = chosen_location
                    conditional_entropy = update_information(conditional_entropy, chosen_location, settings)

                for label in meetings.keys():
                    team[label].first = meetings[label]

                # perform sequential allocation to generate paths, using previous entropy field
                plans = dict()
                for agent in team.values():
                    weights = sn.filters.convolve(conditional_entropy,
                                                  np.ones(settings.image_size),
                                                  mode='constant', cval=0)

                    agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]

                    for location in agent_path:
                        conditional_entropy = update_information(conditional_entropy, location, settings)

                    plans[agent.label] = agent_path

                # update team belief using all observations
                advance = False
                team_observation = set()
                for agent in team.values():
                    # update position
                    agent.position = plans[agent.label][0]

                    # update agent belief
                    _, observation = get_image(agent, sim, settings)
                    team_belief = update_belief(sim.group, team_belief, advance, observation, settings, control=None)

                if t > 1 and (t-1)%settings.process_update == 0:
                    advance = True
                    team_belief = update_belief(sim.group, team_belief, advance, dict(), settings, control=None)

            else:

                for agent in team.values():

                    # predict belief forward (open-loop)
                    predicted_belief = agent.belief
                    belief_updates = settings.meeting_interval//settings.process_update
                    for _ in range(belief_updates):
                        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

                    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
                    conditional_entropy += 0.1

                    weights = sn.filters.convolve(conditional_entropy,
                                                  np.ones(settings.image_size),
                                                  mode='constant', cval=0)

                    # # find reachable locations, and compute the highest weight path to each
                    # distances = np.linalg.norm(node_locations - agent.position, ord=np.inf, axis=2)
                    # locations_r, locations_c = np.where(distances == settings.meeting_interval)
                    # locations = list(zip(locations_r, locations_c))
                    #
                    # options = []
                    # highest_weight = -1
                    # for end in locations:
                    #     _, v = graph_search(agent.position, end, settings.meeting_interval, weights, settings)
                    #
                    #     if v > highest_weight:
                    #         highest_weight = v
                    #     options.append((v, end))
                    #
                    # # pick a location with highest total information gain
                    # options = [end[1] for end in options if end[0] == highest_weight]
                    # meeting = options[0]

                    # find reachable locations, and choose one with high entropy
                    distances = np.linalg.norm(node_locations - agent.position, ord=np.inf, axis=2)
                    locations_r, locations_c = np.where(distances == settings.meeting_interval)
                    locations = list(zip(locations_r, locations_c))

                    if len(locations) == 1:
                        meeting = locations[0]
                    else:
                        # np.random.shuffle(locations)
                        options = [(weights[r, c], (r, c)) for (r, c) in locations]
                        meeting = max(options, key=itemgetter(0))[1]

                    # plan a path to location and update position
                    agent.first = meeting
                    agent.plan = create_solo_plan(agent, sim.group, settings)
                    agent.position = agent.plan[0]

                    # update agent belief
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
            save_data[seed]['coverage'].append(current_coverage)
            # compute_frequency(team, state, frequency)
            # save_data[seed][t] = [compute_accuracy(team[label].belief, state, settings) for label in team.keys()]

        # save_data[seed][total_iterations] = frequency
        print('[Baseline] finished simulation ' + str(sim_count+1) +
              ' (coverage = ' + str(np.mean(save_data[seed]['coverage'])) + ')')

    # write data to file
    filename = 'Benchmark/baseline-'
    if communication:
        filename += 'ycomm-'
    else:
        filename += 'ncomm-'
    filename += 'tau' + str(tau).zfill(2) + 'C' + str(C).zfill(2) + 'pc' + str(pc) + '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    toc = time.clock()
    dt = toc - tic
    print('[Baseline] completed at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    print('[Baseline] %0.2fs = %0.2fm = %0.2fh elapsed' % (dt, dt/60, dt/3600))
