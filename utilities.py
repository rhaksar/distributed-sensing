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
        self.dimension = 25
        self.estimate_process_update = 5
        self.true_process_update = 5

        self.cell_side_length = 0.5

        # cx = np.linspace(0, self.dimension-1, self.dimension) + 0.5
        # cy = np.linspace(0, self.dimension-1, self.dimension) + 0.5
        # Cx, Cy = np.meshgrid(cx, cy)
        # self.Cxy = np.stack([Cx, Cy], axis=2) # .reshape((self.dimension*self.dimension, 2))

        cell_row = np.linspace(0, self.dimension-1, self.dimension)
        cell_col = np.linspace(0, self.dimension-1, self.dimension)
        Cell_row, Cell_col = np.meshgrid(cell_row, cell_col)
        Cell_row, Cell_col = Cell_row.T, Cell_col.T
        self.Crc = np.stack([Cell_row, Cell_col], axis=2)

        self.team_size = 4
        self.image_size = (3, 3)

        # self.deploy_interval = 2
        # self.deploy_locations = (np.array([1.5, 2.5]), np.array([2.5, 1.5]))
        self.meeting_interval = 10
        self.total_interval = None
