from copy import copy
import numpy as np
from operator import itemgetter
from queue import PriorityQueue
import scipy.ndimage as sn
import scipy.stats as ss

from filter import update_belief, measure_model


def schedule_initial_meetings(team, Sprime, cell_locations, config):
    entropy = compute_entropy(team[1].belief, config)

    for i, s in enumerate(Sprime):
        weights = sn.filters.convolve(entropy, np.ones(config.image_size), mode='constant', cval=0)

        distances = np.maximum.reduce([np.linalg.norm(cell_locations-team[k].position, ord=np.inf, axis=2)
                                       for k in s])
        locations_r, locations_c = np.where(distances == config.meeting_interval)
        locations = list(zip(locations_r, locations_c))

        if len(locations) == 1:
            meeting = locations[0]
        else:
            np.random.shuffle(locations)
            options = [(weights[r, c], (r, c)) for (r, c) in locations]
            meeting = max(options, key=itemgetter(0))[1]

        for k in s:
            team[k].last = meeting
            team[k].budget = config.meeting_interval

        for r in range(meeting[0] - config.half_height, meeting[0] + config.half_height + 1):
            for c in range(meeting[1] - config.half_width, meeting[1] + config.half_width + 1):
                if 0 <= r < config.dimension and 0 <= c < config.dimension:
                    entropy[r, c] = 0


def schedule_next_meeting(sub_team, simulation_group, cell_locations, config):
    predicted_belief = copy(sub_team[0].belief)
    belief_updates = config.meeting_interval//config.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(simulation_group, predicted_belief, True, dict(), config)

    conditional_entropy = compute_conditional_entropy(predicted_belief, simulation_group, config)

    for agent in sub_team:
        for location in [agent.first, agent.last]:

            for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
                for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
                    if 0 <= r < config.dimension and 0 <= c < config.dimension:
                        conditional_entropy[r, c] = 0

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    distances = np.maximum.reduce([np.linalg.norm(cell_locations - agent.last, ord=np.inf, axis=2)
                                   for agent in sub_team if agent.first != agent.last])
    locations_r, locations_c = np.where(distances == config.meeting_interval)
    locations = list(zip(locations_r, locations_c))

    if len(locations) == 1:
        meeting = locations[0]

    else:
        np.random.shuffle(locations)
        options = []

        for end in locations:
            v = 0

            for agent in sub_team:
                if agent.first == agent.last:
                    _, w = graph_search(agent.last, end, 2*config.meeting_interval, weights, config)
                else:
                    _, w = graph_search(agent.last, end, config.meeting_interval, weights, config)

                v += w
            v /= len(sub_team)
            options.append((v, end))

        meeting = max(options, key=itemgetter(0))[1]

    for agent in sub_team:
        if agent.first == agent.last:
            agent.first = meeting
            agent.last = meeting
            agent.budget = 2*config.meeting_interval
        else:
            agent.first = copy(agent.last)
            agent.last = meeting
            agent.budget = config.meeting_interval


def create_joint_plan(sub_team, simulation_group, config):
    conditional_entropy = compute_conditional_entropy(sub_team[0].belief, simulation_group, config)

    plans = dict()
    for agent in sub_team:
        plans[agent.label] = []

        weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

        if agent.first == agent.last:
            came_from, _ = graph_search(agent.position, agent.last, 2*config.meeting_interval, weights, config)
            sub_path = get_path(agent.position, agent.last, came_from)

        else:
            sub_path = []
            came_from, _ = graph_search(agent.position, agent.first, config.meeting_interval, weights, config)
            sub_path.extend(get_path(agent.position, agent.first, came_from))
            came_from, _ = graph_search(agent.first, agent.last, config.meeting_interval, weights, config)
            sub_path.extend(get_path(agent.first, agent.last, came_from))

        for location in sub_path:
            for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
                for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
                    if 0 <= r < config.dimension and 0 <= c < config.dimension:
                        conditional_entropy[r, c] = 0

        plans[agent.label].extend(sub_path)

    for agent in sub_team:
        for label in plans.keys():
            if label == agent.label:
                continue
            agent.other_plans[label] = copy(plans[label])


def create_solo_plan(agent, simulation_group, config):
    conditional_entropy = compute_conditional_entropy(agent.belief, simulation_group, config)

    for other_label in agent.other_plans.keys():
        if not agent.other_plans[other_label]:
            continue
        location = agent.other_plans[other_label].pop(0)

        for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
            for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
                if 0 <= r < config.dimension and 0 <= c < config.dimension:
                    conditional_entropy[r, c] = 0

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    came_from, _ = graph_search(agent.position, agent.first, agent.budget, weights, config)
    agent.plan = get_path(agent.position, agent.first, came_from)


def compute_entropy(belief, config):
    entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        entropy[key[0], key[1]] = ss.entropy(belief[key])

    return entropy


def compute_conditional_entropy(belief, simulation_group, config):
    conditional_entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        element = simulation_group[key]
        p_yi_ci = np.asarray([[measure_model(element, ci, yi, config) for ci in element.state_space]
                              for yi in element.state_space])
        p_yi = np.matmul(p_yi_ci, belief[key])
        for yi in element.state_space:
            if p_yi[yi] <= 1e-100:
                continue
            for ci in element.state_space:
                if belief[key][ci] <= 1e-100 or p_yi_ci[yi, ci] <= 1e-100:
                    continue
                conditional_entropy[key[0], key[1]] -= p_yi_ci[yi, ci] * belief[key][ci] * (np.log(p_yi_ci[yi, ci])
                                                                                            + np.log(belief[key][ci])
                                                                                            - np.log(p_yi[yi]))
    return conditional_entropy


def graph_search(start, end, length, weights, config):
    frontier = PriorityQueue()

    start_length = (start, length)
    frontier.put((-weights[start], start_length))

    came_from = dict()
    cost_so_far = dict()
    came_from[start_length] = None
    cost_so_far[start_length] = weights[start]

    while not frontier.empty():
        current = frontier.get()[1]
        current_location = current[0]
        current_length = current[1]

        neighbors = []
        for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_dist = np.linalg.norm(np.asarray(current_location)+np.asarray([dr, dc]) - np.asarray(end), ord=np.inf)
            if new_dist >= current_length:
                continue

            if 0 <= current_location[0]+dr < config.dimension and 0 <= current_location[1]+dc < config.dimension:
                neighbors.append(((current_location[0]+dr, current_location[1]+dc), int(current_length-1)))

        for n in neighbors:
            new_cost = cost_so_far[current] + weights[n[0][0], n[0][1]]
            if n not in cost_so_far or new_cost > cost_so_far[n]:
                cost_so_far[n] = new_cost
                frontier.put((-new_cost, n))
                came_from[n] = current

    return came_from, cost_so_far[(end, 0)]


def get_path(start, end, came_from):
    path = [end]
    current = (end, 0)
    while came_from[current][0] != start:
        previous = came_from[current]
        path.insert(0, previous[0])
        current = previous

    return path
