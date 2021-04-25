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

    def __init__(self, process_update=2, team_size=5, meeting_interval=5, measure_correct=0.95):
        self.seed = None  # random seed
        self.dimension = 25  # size of one side of the square forest

        # create a 2D numpy array which corresponds to the physical location of trees
        cell_row = np.linspace(0, self.dimension-1, self.dimension)
        cell_col = np.linspace(0, self.dimension-1, self.dimension)
        cell_row, cell_col = np.meshgrid(cell_row, cell_col)
        self.cell_locations = np.stack([cell_row.T, cell_col.T], axis=2)

        self.process_update = process_update  # how often forest fire dynamics update

        self.corner = xy_to_rc(self.dimension, np.array([1., 1.]))

        self.team_size = team_size
        self.image_size = (3, 3)
        self.half_height, self.half_width = (self.image_size[0]-1)//2, (self.image_size[1]-1)//2

        self.movements = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

        self.gamma = 0.9  # scale factor for considering "high-value" meeting locations
        self.meeting_interval = meeting_interval  # frequency of meetings

        self.measure_correct = measure_correct  # probability of observing the correct tree state
        self.threshold = 1e-100  # smallest non-zero probability
