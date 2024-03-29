from copy import copy
import numpy as np
from operator import itemgetter
import pickle
import scipy.ndimage as sn
import sys
import time

from framework.filter import update_belief, get_image
from framework.metrics import compute_coverage
from framework.scheduling import create_solo_plan, \
    compute_conditional_entropy, graph_search
from framework.uav import UAV
from framework.utilities import Config

from simulators.fires.LatticeForest import LatticeForest


if __name__ == '__main__':
    print('[Baseline] started at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()

    if len(sys.argv) != 3:
        communication = True
    else:
        communication = bool(int(sys.argv[1]))

    if communication:
        print('[Baseline]     team communication')
    else:
        print('[Baseline]     no communication')

    total_simulations = 10
    offset = 10
    rho = 1
    total_iterations = rho*60 + 1
    tau = 8

    if len(sys.argv) != 3:
        C = 5
    else:
        C = int(sys.argv[2])

    pc = 0.95
    print('[Baseline] rho = ' + str(rho) + ', tau = ' + str(tau) + ', C = ' + str(C) + ', pc = ' + str(pc))

    settings = Config(process_update=rho, team_size=C, meeting_interval=tau, measure_correct=pc)
    square_size = np.ceil(np.sqrt(settings.team_size/2)).astype(int)
    S = []
    for i in range(1, np.floor(settings.team_size/2).astype(int)+1):
        S.append((2*i-1, 2*i))

    # initialize simulator
    sim = LatticeForest(settings.dimension)

    # initialize agents
    initial_belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        # exact belief
        initial_belief[key] = np.zeros(len(element.state_space))
        initial_belief[key][element.state] = 1

    # initialize data structure for saving information
    save_data = dict()

    for sim_count, seed in enumerate(range(offset, total_simulations+offset)):
        np.random.seed(seed)
        sim.rng = seed
        sim.reset()
        save_data[seed] = dict()

        team = {i+1: UAV(label=i+1, belief=copy(initial_belief), image_size=settings.image_size)
                for i in range(settings.team_size)}

        team_belief = None
        if communication:
            team_belief = copy(initial_belief)

        # deploy agents according to schedule S
        for ith_meeting, s in enumerate(S):
            idx = np.unravel_index(ith_meeting, (square_size, square_size), order='C')
            position = (settings.corner[0] - idx[0], settings.corner[1] + idx[1])
            for k in s:
                team[k].position = position
                team[k].budget = settings.meeting_interval

        # deploy remaining agents that do not have a meeting in S
        offset = len(S)+1
        for agent in team.values():
            if not agent.position:
                idx = np.unravel_index(offset, (square_size, square_size), order='C')
                agent.position = (settings.corner[0] - idx[0], settings.corner[1] + idx[1])
                agent.budget = settings.meeting_interval
                offset += 1

        save_data[seed]['coverage'] = []

        # main loop
        for t in range(1, total_iterations):
            # print('time {0:d}'.format(t))

            if communication:
                # print('communication')

                # predict future belief of team belief (open-loop)
                predicted_belief = copy(team_belief)
                belief_updates = settings.meeting_interval//settings.process_update
                for _ in range(belief_updates):
                    predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

                if (t-1) % settings.meeting_interval == 0:
                    # find locations of high entropy and use them as planned locations
                    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
                    conditional_entropy += 0.1

                    for agent in team.values():

                        weights = sn.filters.convolve(conditional_entropy,
                                                      np.ones(settings.image_size),
                                                      mode='constant', cval=0)

                        distances = np.linalg.norm(settings.cell_locations - agent.position, ord=np.inf, axis=2)
                        locations_r, locations_c = np.where(distances == settings.meeting_interval)
                        locations = list(zip(locations_r, locations_c))

                        if len(locations) == 1:
                            chosen_location = locations[0]
                        else:
                            options = []
                            highest_weight = -1
                            for end in locations:
                                _, v = graph_search(agent.position, end, settings.meeting_interval, weights, settings)

                                if v > highest_weight:
                                    highest_weight = v
                                options.append((v, end))

                            options = [end[1] for end in options if end[0] >= settings.gamma*highest_weight]
                            np.random.shuffle(options)
                            chosen_location = options[0]

                        agent.first = chosen_location

                        # conditional_entropy = update_information(conditional_entropy, chosen_location, settings)
                        conditional_entropy[chosen_location[0], chosen_location[1]] = 0

                    # perform sequential allocation to generate paths
                    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
                    conditional_entropy += 0.1

                    for agent in team.values():
                        weights = sn.filters.convolve(conditional_entropy,
                                                      np.ones(settings.image_size),
                                                      mode='constant', cval=0)

                        agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]

                        for location in agent_path:
                            conditional_entropy[location[0], location[1]] = 0

                        agent.plan = agent_path[1:]

                for agent in team.values():
                    agent.position = agent.plan[0]
                    agent.plan.pop(0)

                # update team belief using all observations
                team_observation = dict()
                for agent in team.values():
                    _, observation = get_image(agent, sim, settings)
                    for key in observation.keys():
                        if key not in team_observation:
                            team_observation[key] = []
                        team_observation[key].append(observation[key])

                advance = False
                if t > 1 and (t-1)%settings.process_update == 0:
                    advance = True
                team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

            else:
                # print('no communication')
                for agent in team.values():

                    # predict belief forward (open-loop)
                    predicted_belief = copy(agent.belief)
                    belief_updates = settings.meeting_interval//settings.process_update
                    for _ in range(belief_updates):
                        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

                    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
                    conditional_entropy += 0.1

                    weights = sn.filters.convolve(conditional_entropy,
                                                  np.ones(settings.image_size),
                                                  mode='constant', cval=0)

                    # find reachable locations, and choose one with high entropy
                    distances = np.linalg.norm(settings.cell_locations - agent.position, ord=np.inf, axis=2)
                    locations_r, locations_c = np.where(distances == settings.meeting_interval)
                    locations = list(zip(locations_r, locations_c))

                    if len(locations) == 1:
                        chosen_location = locations[0]
                    else:
                        options = [(weights[r, c], (r, c)) for (r, c) in locations]
                        chosen_location = max(options, key=itemgetter(0))[1]

                    # plan a path to location and update position
                    agent.first = chosen_location
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
            current_coverage = compute_coverage(team, sim, settings)
            save_data[seed]['coverage'].append(current_coverage)

        print('[Baseline] finished simulation {0:d} (coverage = {1:0.4f})'.format(sim_count+1,
                                                                                  np.mean(save_data[seed]['coverage'])))

    # write data to file
    filename = 'Benchmark/baseline-'
    if communication:
        filename += 'ycomm-'
    else:
        filename += 'ncomm-'
    filename += 'rho' + str(rho).zfill(2) + 'tau' + str(tau).zfill(2) + 'C' + str(C).zfill(2) + 'pc' + str(pc) + '.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('[Baseline] mean coverage = {0:0.4f}'.format(np.mean([np.mean(save_data[seed]['coverage'])
                                                                for seed in save_data.keys()])))

    toc = time.clock()
    dt = toc - tic
    print('[Baseline] completed at %s' % (time.strftime('%d-%b-%Y %H:%M')))
    print('[Baseline] %0.2fs = %0.2fm = %0.2fh elapsed' % (dt, dt/60, dt/3600))
