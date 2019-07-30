import numpy as np
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


def set_next_meeting(group, config):
    pass


def graph_search(start, end, weights):
    pass


def create_solo_plan():
    pass


def create_joint_plan():
    pass
