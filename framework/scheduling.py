import networkx as nx
import numpy as np
from operator import itemgetter
import scipy.ndimage as sn
import scipy.stats as ss

from framework.filter import update_belief, measure_model


def schedule_initial_meetings(team, Sprime, simulation_group, cell_locations, config):
    """
    Once the UAVs are deployed, each UAV needs a future meeting location to plan for. This function assigns locations
    for meetings according to the schedule describing by Sprime, such that the location is reachable by each UAV
    given the planning budget. A sequential allocation strategy is used to prevent scheduling multiple meetings close
    together in the forest.

    :param team: a list of UAV objects representing the team.
    :param Sprime: a list of tuples, where each tuple describes a meeting. for example, "(1, 2)" indicates that UAVs 1
    and 2 will meet in the future.
    :param simulation_group: the group of simulation elements from the LatticeForest simulator, which is a dictionary of
    {Tree position: Tree object} key value pairs.
    :param cell_locations: a 2D numpy array which describes the location of trees which encodes the physical dimensions
    of the forest.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: a dictionary of {UAV label: meeting location} key value pairs.
    """

    # compute the conditional entropy for each location in the forest, each UAV has the same belief in the beginning
    # so any UAV's belief can be used
    conditional_entropy = compute_conditional_entropy(team[1].belief, simulation_group, config)
    meetings = dict()

    for s in Sprime:

        # compute a weight for each location in the forest
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

        # update the conditional entropy to account for information gained by UAVs moving to the chosen location,
        # before moving on to the next meeting location
        conditional_entropy = update_information(conditional_entropy, meeting, config)

    return meetings


def schedule_next_meeting(sub_team, merged_belief, simulation_group, cell_locations, config):
    """
    For a sub-group of UAVs, find the next meeting location.

    :param sub_team: a list of UAV objects representing the UAVs planning the next meeting location.
    :param merged_belief: a dictionary representing the merged belief of the sub-group, consisting of
    {Tree position: list of probabilities for each state value} key value pairs.
    :param simulation_group: the group of simulation elements from the LatticeForest simulator, which is a dictionary of
    {Tree position: Tree object} key value pairs.
    :param cell_locations: a 2D numpy array which describes the location of trees which encodes the physical dimensions
    of the forest.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: a tuple indicating the chosen meeting location in the forest.
    """

    # perform an open-loop prediction of the belief for the next meeting
    predicted_belief = merged_belief
    belief_updates = config.meeting_interval//config.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(simulation_group, predicted_belief, True, dict(), config)

    conditional_entropy = compute_conditional_entropy(predicted_belief, simulation_group, config)

    # account for intermediate meetings that occur before the sub-group of UAVs meets again, which reduces the
    # information that could be gained due to other UAVs making observations
    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, config)

        conditional_entropy = update_information(conditional_entropy, agent.first, config)

    # compute a weight for each location in the forest
    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    # the first and last UAV in the team only ever have one meeting, whereas other UAVs have two meetings
    # find the reachable locations for the UAVs in the sub-group based on their meeting locations
    if all([agent.first == agent.last for agent in sub_team]):
        distances = np.maximum.reduce([np.linalg.norm(cell_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team])
    else:
        distances = np.maximum.reduce([np.linalg.norm(cell_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team if agent.label not in [1, config.team_size]])
    locations_r, locations_c = np.where(distances == config.meeting_interval)
    locations = list(zip(locations_r, locations_c))

    if len(locations) == 1:
        meeting = locations[0]

    # if there's more than one possible meeting location, compute the highest-weight path each UAV could take for each
    # location. average the highest-weight paths to produce a value for each location, and choose a location with a
    # high value
    else:
        options = []
        highest_weight = -1

        for end in locations:
            v = 0

            for agent in sub_team:
                if agent.label in [1, config.team_size]:
                    _, w = graph_search(agent.last, end, 2*config.meeting_interval, weights, config)
                else:
                    _, w = graph_search(agent.last, end, config.meeting_interval, weights, config)

                v += w
            v /= len(sub_team)

            if v > highest_weight:
                highest_weight = v
            options.append((v, end))

        # options = [end[1] for end in options if end[0] >= 0.9*highest_weight]
        # randomize the location based on high-weight locations, to prevent inadvertently scheduling many meetings in
        # the same location, since a given sub-group is unaware of what other sub-groups are planning
        options = [end for (weight, end) in options if weight >= config.gamma*highest_weight]
        np.random.shuffle(options)
        meeting = options[0]

    return meeting


def create_joint_plan(sub_team, simulation_group, config):
    """
    Compute paths for a team of UAVs to reach a meeting within the planning budget, using a sequential allocation
    strategy.

    :param sub_team: a list of UAV objects representing the UAVs planning the next meeting location.
    :param simulation_group: the group of simulation elements from the LatticeForest simulator, which is a dictionary of
    {Tree position: Tree object} key value pairs.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: a dictionary of {UAV.label: list of positions} key value pairs, describing the trajectories for each UAV in
    the sub-group.
    """
    conditional_entropy = compute_conditional_entropy(sub_team[0].belief, simulation_group, config)

    # account for intermediate meetings that occur before the sub-group of UAVs meets again, which reduces the
    # information that could be gained due to other UAVs making observations
    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, config)

    # sequential allocation strategy: plan path for a UAV, account for information gained by path, repeat for next UAV
    plans = dict()
    for agent in sub_team:
        weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)
        plans[agent.label] = []

        # first and last UAV in team only have one meeting, all other UAVs have two meetings
        if agent.label in [1, config.team_size]:
            sub_path = graph_search(agent.position, agent.last, 2*config.meeting_interval, weights, config)[0]

        else:
            sub_path = []
            sub_path.extend(graph_search(agent.position, agent.first, config.meeting_interval, weights, config)[0])
            sub_path.extend(graph_search(agent.first, agent.last, config.meeting_interval, weights, config)[0])

        # subtract information gained by path
        for location in sub_path:
            conditional_entropy = update_information(conditional_entropy, location, config)

        plans[agent.label].extend(sub_path)

    return plans


def create_solo_plan(agent, simulation_group, config):
    """
    Create a path for a UAV starting at its current position and ending at its next meeting.

    :param agent: a UAV class object.
    :param simulation_group: the group of simulation elements from the LatticeForest simulator, which is a dictionary of
    {Tree position: Tree object} key value pairs.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: a list of locations describing the planned path.
    """

    # account for the information that other UAVs will gain as they follow their planned paths
    conditional_entropy = compute_conditional_entropy(agent.belief, simulation_group, config)

    for other_label in agent.other_plans.keys():
        if not agent.other_plans[other_label]:
            continue
        location = agent.other_plans[other_label][0]
        conditional_entropy = update_information(conditional_entropy, location, config)

    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    return graph_search(agent.position, agent.first, agent.budget, weights, config)[0][1:]


def compute_entropy(belief, config):
    """
    Compute the entropy of a belief.

    :param belief: a dictionary describing the belief, as {Tree position: list of probabilities for each state} key
    value pairs.
    :param config: a Config class object which contains the meeting and planning parameters.

    :return: the entropy of the belief as a 2D numpy array, where each (row, col) corresponds to a Tree position
    """
    entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        entropy[key[0], key[1]] = ss.entropy(belief[key])

    return entropy


def compute_conditional_entropy(belief, simulation_group, config, offset=0.1):
    """
    Compute the conditional entropy, which is based on a belief and the measurement model.

    :param belief:
    :param simulation_group:
    :param config:
    :param offset: a small positive number to add to the conditional entropy to prevent cases where it is zero for all
    locations in the forest, e.g., when the ground truth is known.

    :return: a 2D numpy representing the conditional entropy over the forest.
    """

    conditional_entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        element = simulation_group[key]
        p_yi_ci = np.asarray([[measure_model(element, ci, yi, config) for ci in element.state_space]
                              for yi in element.state_space])
        p_yi = np.matmul(p_yi_ci, belief[key])
        for yi in element.state_space:
            if p_yi[yi] <= config.threshold:
                continue
            for ci in element.state_space:
                if belief[key][ci] <= config.threshold or p_yi_ci[yi, ci] <= config.threshold:
                    continue
                conditional_entropy[key[0], key[1]] -= p_yi_ci[yi, ci] * belief[key][ci] * (np.log(p_yi_ci[yi, ci])
                                                                                            + np.log(belief[key][ci])
                                                                                            - np.log(p_yi[yi]))
    return conditional_entropy + offset


def graph_search(start, end, length, weights, config):
    """
    Find a highest-weight path from start to end of a given length by constructing a directed acyclic graph
    and running the Bellman-Ford algorithm.

    :param start: tuple describing the start position.
    :param end: tuple describing the start position
    :param length: total path length.
    :param weights: 2D numpy array representing the value of moving to a location in the forest.
    :param config: a Config class object with valid movements for UAVs and the dimension of the square forest.

    :return: a list of positions representing the longest path, and the path weight.
    """

    if max(np.abs(start[0]-end[0]), np.abs(start[1]-end[1])) > length:
        raise Exception('length {0} too short to plan path from start {1} to end {2}'.format(length, start, end))

    graph = nx.DiGraph()
    nodes = [(start, length)]

    # for each node, find other nodes that can be moved to with the remaining amount of path length
    while nodes:
        current_node, current_length = nodes.pop(0)
        if current_length == 0:
            continue

        for (dr, dc) in config.movements:
            neighbor_node = (current_node[0] + dr, current_node[1] + dc)

            if max(abs(current_node[0]+dr-end[0]), abs(current_node[1]+dc-end[1])) >= current_length:
                continue

            neighbor = (neighbor_node, int(current_length-1))
            edge = ((current_node, current_length), neighbor)
            if graph.has_edge(edge[0], edge[1]):
                continue

            if 0 <= neighbor_node[0] < config.dimension and 0 <= neighbor_node[1] < config.dimension:
                nodes.append(neighbor)
                graph.add_edge(edge[0], edge[1], weight=1e-4+weights[neighbor_node[0], neighbor_node[1]])

    if len(graph.edges()) == 1:
        return [start, end], weights[end[0], end[1]]

    path = nx.algorithms.dag_longest_path(graph)
    path_weight = sum([graph.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path)-1)])
    path = [element[0] for element in path]

    return path, path_weight


def update_information(metric, location, config):
    """
    Given a metric represented as a 2D numpy array, zero out at a given location and neighboring locations based on the
    camera field of view.

    :param metric: a 2D numpy array representing a metric over the forest.
    :param location: a location within the forest to zero out the metric.
    :param config: a Config class object containing the image field of view dimensions and the dimension of the square
    forest.

    :return: the modified metric as a 2D numpy array.
    """

    for r in range(location[0] - config.half_height, location[0] + config.half_height + 1):
        for c in range(location[1] - config.half_width, location[1] + config.half_width + 1):
            if 0 <= r < config.dimension and 0 <= c < config.dimension:
                metric[r, c] = 0

    return metric
