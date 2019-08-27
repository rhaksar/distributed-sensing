from copy import copy
import networkx as nx
import numpy as np
from operator import itemgetter
from queue import PriorityQueue
import scipy.ndimage as sn
import scipy.stats as ss
import time

from filter import update_belief, measure_model


def schedule_initial_meetings(team, Sprime, simulation_group, cell_locations, config):
    conditional_entropy = compute_conditional_entropy(team[1].belief, simulation_group, config)
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

    for agent in sub_team:
        for location in [agent.first, agent.last]:
            conditional_entropy = update_information(conditional_entropy, location, config)

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
                    random_weights = np.random.rand(*weights.shape)
                    t0 = time.time()
                    came_from, w1 = graph_search(agent.last, end, 2*config.meeting_interval, random_weights, config)
                    P1 = get_path(agent.last, end, came_from)
                    t1 = time.time()
                    print(t1-t0)
                    t0 = time.time()
                    P2, w2 = graph_search_nx(agent.last, end, 2*config.meeting_interval, random_weights, config)
                    t1 = time.time()
                    print(t1-t0)
                    print('stop here')
                else:
                    _, w = graph_search(agent.last, end, config.meeting_interval, weights, config)

                v += w
            v /= len(sub_team)
            options.append((v, end))

        meeting = max(options, key=itemgetter(0))[1]

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

    for other_label in agent.other_plans.keys():
        if not agent.other_plans[other_label]:
            continue
        location = agent.other_plans[other_label].pop(0)
        conditional_entropy = update_information(conditional_entropy, location, config)

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    came_from, _ = graph_search(agent.position, agent.first, agent.budget, weights, config)
    # agent.plan = get_path(agent.position, agent.first, came_from)

    return get_path(agent.position, agent.first, came_from)


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


def graph_search_nx(start, end, length, weights, config):
    G = nx.DiGraph()
    neighbors = [(start, length)]
    nodes = []

    while neighbors:
        current_node, current_length = neighbors.pop(0)

        for (dr, dc) in config.movements:
            neighbor_node = (current_node[0] + dr, current_node[1] + dc)

            new_dist = np.linalg.norm(np.asarray(current_node) + np.asarray([dr, dc]) - np.asarray(end), ord=np.inf)
            if new_dist >= current_length:
                continue

            neighbor = (neighbor_node, int(new_dist))

            # if neighbor in nodes:
            #     continue

            if 0 <= neighbor_node[0] < config.dimension and 0 <= neighbor_node[1] < config.dimension:
                neighbors.append(neighbor)
                nodes.append(neighbor)
                G.add_edge((current_node, current_length), (neighbor_node, int(new_dist)),
                           weight=-weights[neighbor_node[0], neighbor_node[1]])

    print(nx.is_directed_acyclic_graph(G))
    # return nx.algorithms.dag_longest_path(G), nx.algorithms.dag_longest_path_length(G)
    return nx.algorithms.shortest_path(G, (start, length), (end, 0), weight='weight'), \
           nx.algorithms.shortest_path_length(G, (start, length), (end, 0), weight='weight')


def graph_search(start, end, length, weights, config):
    frontier = PriorityQueue()

    start_node = (start, length)
    frontier.put((-weights[start], start_node))

    came_from = dict()
    cost_so_far = dict()
    came_from[start_node] = None
    cost_so_far[start_node] = weights[start]

    while not frontier.empty():
        current = frontier.get()[1]
        current_location = current[0]
        current_length = current[1]

        neighbors = []
        for (dr, dc) in config.movements:
            new_dist = np.linalg.norm(np.asarray(current_location)+np.asarray([dr, dc]) - np.asarray(end), ord=np.inf)
            # if new_dist >= length:
            #     continue
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


def update_information(metric, location, config):
    for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
        for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
            if 0 <= r < config.dimension and 0 <= c < config.dimension:
                metric[r, c] = 0

    return metric
