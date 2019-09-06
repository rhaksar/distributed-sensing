# from collections import defaultdict
# from copy import copy
# import graph_tool as gt
# import graph_tool.search as gts
# import heapq
import networkx as nx
import numpy as np
from operator import itemgetter
# from queue import PriorityQueue
import scipy.ndimage as sn
import scipy.stats as ss
import time

from filter import update_belief, measure_model


def schedule_initial_meetings(team, Sprime, simulation_group, cell_locations, config):
    conditional_entropy = compute_conditional_entropy(team[1].belief, simulation_group, config)
    conditional_entropy += 0.1
    meetings = dict()

    for s in Sprime:

        weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

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
            meetings[k] = meeting
            # team[k].last = meeting
            # team[k].budget = config.meeting_interval

        conditional_entropy = update_information(conditional_entropy, meeting, config)

    return meetings


def schedule_next_meeting(sub_team, merged_belief, simulation_group, cell_locations, config):
    predicted_belief = merged_belief
    belief_updates = config.meeting_interval//config.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(simulation_group, predicted_belief, True, dict(), config)

    conditional_entropy = compute_conditional_entropy(predicted_belief, simulation_group, config)
    conditional_entropy += 0.1

    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, config)

        # for location in [agent.first, agent.last]:
        #     conditional_entropy = update_information(conditional_entropy, location, config)
        conditional_entropy = update_information(conditional_entropy, agent.first, config)

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    if all([agent.first == agent.last for agent in sub_team]):
        distances = np.maximum.reduce([np.linalg.norm(cell_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team])
    else:
        distances = np.maximum.reduce([np.linalg.norm(cell_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team if agent.first != agent.last])
    locations_r, locations_c = np.where(distances == config.meeting_interval)
    locations = list(zip(locations_r, locations_c))
    # print('valid meeting locations', locations)

    if len(locations) == 1:
        meeting = locations[0]

    else:
        # np.random.shuffle(locations)
        options = []

        highest_weight = -1
        # meeting = None

        for end in locations:
            v = 0

            for agent in sub_team:
                if agent.first == agent.last:
                    _, w = graph_search(agent.last, end, 2*config.meeting_interval, weights, config)
                else:
                    _, w = graph_search(agent.last, end, config.meeting_interval, weights, config)

                v += w
            v /= len(sub_team)

            if v > highest_weight:
                # meeting = end
                highest_weight = v
            options.append((v, end))

        # np.random.shuffle(options)
        # print('all options', options)
        options = [end[1] for end in options if end[0] >= 0.9*highest_weight]
        np.random.shuffle(options)
        meeting = options[0]
        # print('best meeting locations', options)
        # meeting = random.choice(options)
        # meeting = max(options, key=itemgetter(0))[1]

    # for agent in sub_team:
    #     if agent.first == agent.last:
    #         agent.first = meeting
    #         agent.last = meeting
    #         agent.budget = 2*config.meeting_interval
    #     else:
    #         agent.first = copy(agent.last)
    #         agent.last = meeting
    #         agent.budget = config.meeting_interval

    return meeting


def create_joint_plan(sub_team, simulation_group, config):
    conditional_entropy = compute_conditional_entropy(sub_team[0].belief, simulation_group, config)
    conditional_entropy += 0.1

    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, config)

        # for location in [agent.first, agent.last]:
        #     conditional_entropy = update_information(conditional_entropy, location, config)

    plans = dict()
    for agent in sub_team:
        weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)
        plans[agent.label] = []

        if agent.first == agent.last:
            # came_from, _ = graph_search(agent.position, agent.last, 2*config.meeting_interval, weights, config)
            # sub_path = get_path(agent.position, agent.last, came_from)
            sub_path = graph_search(agent.position, agent.last, 2*config.meeting_interval, weights, config)[0]

        else:
            sub_path = []
            # came_from, _ = graph_search(agent.position, agent.first, config.meeting_interval, weights, config)
            # sub_path.extend(get_path(agent.position, agent.first, came_from))
            # came_from, _ = graph_search(agent.first, agent.last, config.meeting_interval, weights, config)
            # sub_path.extend(get_path(agent.first, agent.last, came_from))

            sub_path.extend(graph_search(agent.position, agent.first, config.meeting_interval, weights, config)[0])
            sub_path.extend(graph_search(agent.first, agent.last, config.meeting_interval, weights, config)[0])

        for location in sub_path:
            conditional_entropy = update_information(conditional_entropy, location, config)

        plans[agent.label].extend(sub_path)

    # for agent in sub_team:
    #     for label in plans.keys():
    #         if label == agent.label:
    #             continue
    #         agent.other_plans[label] = copy(plans[label])

    return plans


def create_solo_plan(agent, simulation_group, config):
    conditional_entropy = compute_conditional_entropy(agent.belief, simulation_group, config)
    conditional_entropy += 0.1

    for other_label in agent.other_plans.keys():
        if not agent.other_plans[other_label]:
            continue
        location = agent.other_plans[other_label][0]
        conditional_entropy = update_information(conditional_entropy, location, config)

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    # came_from, _ = graph_search(agent.position, agent.first, agent.budget, weights, config)
    # agent.plan = get_path(agent.position, agent.first, came_from)

    # return get_path(agent.position, agent.first, came_from)
    return graph_search(agent.position, agent.first, agent.budget, weights, config)[0][1:]


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
    if max(np.abs(start[0]-end[0]), np.abs(start[1]-end[1])) > length:
        raise Exception('length {0} too short to plan path from start {1} to end {2}'.format(length, start, end))
    # t0 = time.time()
    graph = nx.DiGraph()
    nodes = [(start, length)]

    while nodes:
        current_node, current_length = nodes.pop(0)
        if current_length == 0:
            continue

        for (dr, dc) in config.movements:
            neighbor_node = (current_node[0] + dr, current_node[1] + dc)

            # new_dist = np.linalg.norm(np.asarray(current_node) + np.asarray([dr, dc]) - np.asarray(end), ord=np.inf)
            if max(abs(current_node[0]+dr-end[0]), abs(current_node[1]+dc-end[1])) >= current_length:
                continue

            neighbor = (neighbor_node, int(current_length-1))
            edge = ((current_node, current_length), neighbor)
            if graph.has_edge(edge[0], edge[1]):
                continue

            if 0 <= neighbor_node[0] < config.dimension and 0 <= neighbor_node[1] < config.dimension:

                # if graph.has_edge(edge[0], edge[1]):
                #     continue
                nodes.append(neighbor)
                graph.add_edge(edge[0], edge[1], weight=1e-4+weights[neighbor_node[0], neighbor_node[1]])

    # t1 = time.time()
    # print(t1-t0)

    if len(graph.edges()) == 1:
        return (start, end), weights[end[0], end[1]]

    path = nx.algorithms.dag_longest_path(graph)
    path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
    path = [element[0] for element in path]

    # if not path:
    #     print(path)
    #     print('failed to plan path from', start, 'to', end, 'with budget', length)
    #     raise Exception()

    return path, path_weight


def update_information(metric, location, config):
    for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
        for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
            if 0 <= r < config.dimension and 0 <= c < config.dimension:
                metric[r, c] = 0

    # rows = range(max(0, location[0]-config.half_height), min(metric.shape[0], location[0]+config.half_height+1))
    # cols = range(max(0, location[1]-config.half_width), min(metric.shape[1], location[1]+config.half_width+1))
    # metric[rows, cols] = 0

    return metric
