from copy import copy
import numpy as np
from operator import itemgetter
from queue import PriorityQueue
import scipy.ndimage as sn
import scipy.stats as ss

from filter import update_belief, measure_model


# def set_initial_meetings(team, schedule, config):
#     taken_positions = []
#     for meeting in schedule[1]:
#         distances = np.maximum.reduce([np.linalg.norm(config.Cxy-team[i].position, ord=np.inf, axis=2)
#                                        for i in meeting])
#         distances[distances > 1*config.meeting_interval] = 0
#         for (r, c) in taken_positions:
#             distances[r, c] = 0
#
#         meet_position_rc = np.unravel_index(distances.argmax(), distances.shape)
#         meet_position = config.Cxy[meet_position_rc[0], meet_position_rc[1], :]
#         # meet_position = np.asarray(rc_to_xy(config.dimension, meet_position_rc)) + config.cell_side_length
#         [team[i].meetings.append([meet_position, config.meeting_interval]) for i in meeting]
#
#         taken_positions.append(meet_position_rc)
#         half_row = (config.image_size[0]-1)//2
#         half_col = (config.image_size[1]-1)//2
#         for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
#             for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
#                 r = meet_position_rc[0] + dr
#                 c = meet_position_rc[1] + dc
#
#                 if 0 <= r < config.dimension and 0 <= c < config.dimension:
#                     taken_positions.append((r, c))
#
#     return

def set_initial_meetings(team, schedule, simulation_group, config):
    predicted_belief = copy(team[1].belief)
    entropy = np.zeros((config.dimension, config.dimension))
    for key in predicted_belief.keys():
        entropy[key[0], key[1]] = ss.entropy(predicted_belief[key])

    half_row = (config.image_size[0]-1)//2
    half_col = (config.image_size[1]-1)//2
    for i in range(1, len(schedule)):

        weights = sn.filters.convolve(entropy, np.ones(config.image_size), mode='constant', cval=0)
        for idx, meeting in enumerate(schedule[i]):
            last_positions = []
            for label in meeting:
                last_positions.append(team[label].meetings[i-1][0])  # this might error, need to fix
            distances = np.maximum.reduce([np.linalg.norm(config.Crc - last_positions[i], ord=np.inf, axis=2)
                                           for i in range(len(last_positions))])
            meeting_r, meeting_c = np.where(distances == config.meeting_interval)
            if (idx+1)%2 == 0:
                meeting_r, meeting_c = np.flip(meeting_r), np.flip(meeting_c)
            options = [(weights[r, c], (r, c)) for (r, c) in zip(meeting_r, meeting_c)]
            best_option = max(options, key=itemgetter(0))[1]
            [team[label].meetings.append([best_option, config.meeting_interval]) for label in meeting]

            for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                    r = best_option[0] + dr
                    c = best_option[1] + dc

                    if 0 <= r < config.dimension and 0 <= c < config.dimension:
                        entropy[r, c] = 0

    return


def set_next_meeting(sub_team, simulation_group, config):
    predicted_belief = copy(sub_team[0].belief)
    belief_updates = (config.total_interval-config.meeting_interval)//config.estimate_process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(simulation_group, predicted_belief, True, dict())

    # current_time = sub_team[0].time
    # for t in range(config.total_interval-config.meeting_interval):
    #     advance = False
    #     if (current_time+t)%config.process_update==0:
    #         advance = True
    #     predicted_belief = update_belief(simulation_group, predicted_belief, advance, dict())

    entropy = np.zeros((config.dimension, config.dimension))
    for key in predicted_belief.keys():
        entropy[key[0], key[1]] = ss.entropy(predicted_belief[key])
    weights = sn.filters.convolve(entropy, np.ones(config.image_size), mode='constant', cval=0)

    last_positions = []
    for agent in sub_team:
        # if not agent.meetings:
        #     continue
        last_positions.append(agent.meetings[-1][0])  # this might error, need to fix

    distances = np.maximum.reduce([np.linalg.norm(config.Crc - last_positions[i], ord=np.inf, axis=2)
                                   for i in range(len(last_positions))])
    meeting_r, meeting_c = np.where(distances == config.meeting_interval)
    options = []
    for (m_r, m_c) in zip(meeting_r, meeting_c):
        # end = xy_to_rc(config.dimension, config.Cxy[m_r, m_c])
        end = (m_r, m_c)
        score = 0
        for position in last_positions:
            # position_rc = xy_to_rc(config.dimension, position)
            _, cost_so_far = graph_search((position, config.meeting_interval), end, -weights, config)
            score += cost_so_far[(end, 0)]
        score /= len(last_positions)
        options.append((score, end))
    best_option = min(options, key=itemgetter(0))[1]
    # best_option = np.asarray(rc_to_xy(config.dimension, best_option_rc)) + config.cell_side_length
    [agent.meetings.append([best_option, config.meeting_interval]) for agent in sub_team]

    # distances[distances != config.meeting_interval] = -1
    # weights = np.multiply(entropy, distances)
    # meet_position_rc = np.unravel_index(weights.argmax(), weights.shape)
    # meet_position = np.asarray(rc_to_xy(config.dimension, meet_position_rc)) + config.cell_side_length
    # [agent.meetings.append([meet_position, config.meeting_interval]) for agent in sub_team]
    return


def create_solo_plan(uav, simulation_group, config):
    belief = uav.belief
    conditional_entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        element = simulation_group[key]
        p_yi_ci = np.asarray([[measure_model(element, ci, yi) for ci in element.state_space]
                              for yi in element.state_space])
        p_yi = np.matmul(p_yi_ci, belief[key])
        for yi in element.state_space:
            for ci in element.state_space:
                conditional_entropy[key[0], key[1]] -= p_yi_ci[yi, ci]*belief[key][ci]*(np.log(p_yi_ci[yi, ci])
                                                                                        + np.log(belief[key][ci])
                                                                                        - np.log(p_yi[yi]))

    half_row = config.image_size[0]//2
    half_col = config.image_size[1]//2
    conditional_entropy = np.pad(conditional_entropy, config.image_size, 'constant', constant_values=(0, 0))
    for other_label in uav.other_plans.keys():
        # if other_label > uav.label:
        #     continue
        location = uav.other_plans[other_label].pop(0)

        coeff = np.zeros_like(conditional_entropy)
        row_start = config.image_size[0]+location[0]-half_row
        row_end = config.image_size[0]+location[0]+half_row+1
        col_start = config.image_size[1]+location[1]-half_col
        col_end = config.image_size[1]+location[1]+half_col+1
        coeff[row_start:row_end, col_start:col_end] = np.ones(config.image_size)
        conditional_entropy -= coeff*conditional_entropy

    conditional_entropy = conditional_entropy[config.image_size[0]:-config.image_size[0],
                                              config.image_size[1]:-config.image_size[1]]
    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    came_from, _ = graph_search((uav.position, uav.meetings[0][1]), uav.meetings[0][0], -weights, config)
    actions = []
    path = [uav.meetings[0][0]]
    current = (uav.meetings[0][0], 0)
    while came_from[current][0] != uav.position:
        previous = came_from[current]
        path.insert(0, previous[0])
        actions.insert(0, (current[0][0]-previous[0][0], current[0][1]-previous[0][1]))
        current = previous

    # uav.meetings[0][1] -= 1
    # uav.next_position = (uav.position[0]+actions[0][0], uav.position[1]+actions[0][1])
    return path


def create_joint_plan(sub_team, simulation_group, config):
    belief = sub_team[0].belief
    # entropy = np.zeros((config.dimension, config.dimension))
    # for key in belief.keys():
    #     entropy[key[0], key[1]] = ss.entropy(belief[key])
    conditional_entropy = np.zeros((config.dimension, config.dimension))
    for key in belief.keys():
        element = simulation_group[key]
        p_yi_ci = np.asarray([[measure_model(element, ci, yi) for ci in element.state_space]
                              for yi in element.state_space])
        p_yi = np.matmul(p_yi_ci, belief[key])
        for yi in element.state_space:
            for ci in element.state_space:
                conditional_entropy[key[0], key[1]] -= p_yi_ci[yi, ci]*belief[key][ci]*(np.log(p_yi_ci[yi, ci])
                                                                                        + np.log(belief[key][ci])
                                                                                        - np.log(p_yi[yi]))
    weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

    half_row = config.image_size[0]//2
    half_col = config.image_size[1]//2
    plans = dict()
    agent_labels = [(agent.label, agent) for agent in sub_team]
    agent_labels.sort(key=itemgetter(0))
    for (label, agent) in agent_labels:
        plans[agent.label] = []
        # start = xy_to_rc(config.dimension, agent.position)
        start = agent.position

        for meeting in agent.meetings:
            # end = xy_to_rc(config.dimension, meeting[0])
            end = meeting[0]
            came_from, _ = graph_search((start, config.meeting_interval), end, -weights, config)
            sub_path = [end]
            current = (end, 0)
            while came_from[current][0] != start:
                previous = came_from[current]
                sub_path.insert(0, previous[0])
                current = previous

            conditional_entropy = np.pad(conditional_entropy, config.image_size, 'constant', constant_values=(0, 0))
            for location in sub_path:
                coeff = np.zeros_like(conditional_entropy)
                row_start = config.image_size[0]+location[0]-half_row
                row_end = config.image_size[0]+location[0]+half_row+1
                col_start = config.image_size[1]+location[1]-half_col
                col_end = config.image_size[1]+location[1]+half_col+1
                coeff[row_start:row_end, col_start:col_end] = np.ones(config.image_size)
                conditional_entropy -= coeff*conditional_entropy
            conditional_entropy = conditional_entropy[config.image_size[0]:-config.image_size[0],
                                                      config.image_size[1]:-config.image_size[1]]
            weights = sn.filters.convolve(conditional_entropy, np.ones(config.image_size), mode='constant', cval=0)

            plans[agent.label].extend(sub_path)
            start = end

    for agent in sub_team:
        for label in plans.keys():
            if label == agent.label:
                continue
            agent.other_plans[label] = copy(plans[label])
        # agent.other_plans = copy(plans)
        # agent.other_plans.pop(agent.label)

    return


def graph_search(start, end, weights, config):
    frontier = PriorityQueue()
    frontier.put((0, start))

    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()[1]
        current_location = current[0]
        current_budget = current[1]

        neighbors = []
        for (dr, dc) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_dist = np.linalg.norm(np.asarray(current_location)+np.asarray([dr, dc]) - np.asarray(end), ord=np.inf)
            if new_dist >= current_budget:
                continue

            if 0 <= current_location[0]+dr < config.dimension and 0 <= current_location[1]+dc < config.dimension:
                neighbors.append(((current_location[0]+dr, current_location[1]+dc), int(current_budget-1)))

        for n in neighbors:
            new_cost = cost_so_far[current] + weights[n[0][0], n[0][1]]
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                priority = new_cost
                frontier.put((priority, n))
                came_from[n] = current

    return came_from, cost_so_far
