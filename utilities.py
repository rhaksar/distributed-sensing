import numpy as np


def col_to_x(col):
    return 1*col


def row_to_y(row_limit, row):
    return 1*(row_limit-row-1)


def rc_to_xy(row_limit, rc):
    return col_to_x(rc[1]), row_to_y(row_limit, rc[0])


def x_to_col(x):
    return np.floor(x).astype(np.int8)


def y_to_row(y_limit, y):
    return np.ceil(y_limit-1-y).astype(np.int8)


def xy_to_rc(y_limit, xy):
    return y_to_row(y_limit, xy[1]), x_to_col(xy[0])


class Config(object):

    def __init__(self):
        self.seed = None
        self.dimension = 25

        self.process_update = 10
        self.regularization_weight = 0

        self.cell_side_length = 0.5
        self.corner = xy_to_rc(self.dimension, np.array([1.5, 1.5]) - self.cell_side_length)

        self.team_size = 5
        self.image_size = (3, 3)
        self.half_height, self.half_width = (self.image_size[0]-1)//2, (self.image_size[1]-1)//2

        # self.deploy_interval = 2
        # self.deploy_locations = (np.array([1.5, 2.5]), np.array([2.5, 1.5]))
        self.meeting_interval = 7

        self.measure_correct = 0.95
