from copy import copy
from filter import update_belief
import numpy as np
import scipy.stats as ss
from utilities import rc_to_xy

def set_initial_meetings(team, schedule, config):
    taken_positions = []
    for meeting in schedule[1]:
        distances = np.maximum.reduce([np.linalg.norm(config.Cxy-team[i].position, ord=np.inf, axis=2)
                                       for i in meeting])
        distances[distances > 1*config.meeting_interval] = 0
        for (r, c) in taken_positions:
            distances[r, c] = 0

        meet_position_rc = np.unravel_index(distances.argmax(), distances.shape)
        meet_position = np.asarray(rc_to_xy(config.dimension, meet_position_rc)) + config.cell_side_length
        [team[i].meetings.append([meet_position, config.meeting_interval]) for i in meeting]

        taken_positions.append(meet_position_rc)
        half_row = (config.image_size[0]-1)//2
        half_col = (config.image_size[1]-1)//2
        for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
            for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                r = meet_position_rc[0] + dr
                c = meet_position_rc[1] + dc

                if 0 <= r < config.dimension and 0 <= c < config.dimension:
                    taken_positions.append((r, c))

    return


def set_next_meeting(sub_team, simulation_group, config):
    predicted_belief = copy(sub_team[0].belief)
    current_time = copy(sub_team[0].time)
    belief_updates = (config.total_interval-config.meeting_interval)//config.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(simulation_group, predicted_belief, True, dict())
    # for t in range(config.total_interval-config.meeting_interval):
    #     advance = False
    #     if (current_time+t)%config.process_update==0:
    #         advance = True
    #     predicted_belief = update_belief(simulation_group, predicted_belief, advance, dict())

    entropy = np.zeros((config.dimension, config.dimension))
    for key in predicted_belief.keys():
        entropy[key[0], key[1]] = ss.entropy(predicted_belief[key])

    last_positions = []
    for agent in sub_team:
        if not agent.meetings:
            continue
        last_positions.append(agent.meetings[-1][0])

    distances = np.maximum.reduce([np.linalg.norm(config.Cxy - last_positions[i], ord=np.inf, axis=2)
                                   for i in range(len(last_positions))])
    distances[distances != config.meeting_interval] = -1
    weights = np.multiply(entropy, distances)
    meet_position_rc = np.unravel_index(weights.argmax(), weights.shape)
    meet_position = np.asarray(rc_to_xy(config.dimension, meet_position_rc)) + config.cell_side_length
    [agent.meetings.append([meet_position, config.meeting_interval]) for agent in sub_team]
    return


def create_solo_plan(uav):
    pass


def create_joint_plan(sub_team):
    pass


def graph_search(start, end, weights):
    pass
