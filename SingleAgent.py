from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import os
import scipy.ndimage as sn
import scipy.stats as ss
import sys

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')

from filter import ApproxFilter, multiply_probabilities
from fires.UrbanForest import UrbanForest
from fires.ForestElements import Tree, SimpleUrban


def observation_probability(element, state, observation):
    measure_correct = 0.9
    measure_wrong = None
    if isinstance(element, Tree):
        measure_wrong = 0.5*(1-measure_correct)
    elif isinstance(element, SimpleUrban):
        measure_wrong = (1/3)*(1-measure_correct)

    if state != observation:
        return measure_wrong
    elif state == observation:
        return measure_correct


def filter_element_update(key, element, prior, advance, control, observation):
    posterior = np.zeros(len(element.state_space))
    num_neighbors = len(element.neighbors)
    on_fire = 1

    caf = np.zeros(num_neighbors+1)
    for l in range(2**num_neighbors):
        xj = np.base_repr(l, base=2).zfill(num_neighbors)
        active = xj.count('1')

        values = []
        for n in range(num_neighbors):
            neighbor_key = element.neighbors[n]
            prob = None
            if int(xj[n]) == 0:
                prob = 1 - prior[neighbor_key][on_fire]
            elif int(xj[n]) == 1:
                prob = prior[neighbor_key][on_fire]

            values.append(prob)

        caf[active] += multiply_probabilities(values)

    for s_t in element.state_space:
        for s_tm1 in element.state_space:
            if advance:
                for active in range(num_neighbors+1):
                    values = [element.dynamics((s_tm1, active, s_t), control[key]), caf[active], prior[key][s_tm1]]
                    posterior[s_t] += multiply_probabilities(values)
            else:
                posterior[s_t] += (s_t == s_tm1)*prior[key][s_tm1]

        if key in observation.keys():
            posterior[s_t] *= observation_probability(element, s_t, observation[key])

    posterior /= np.sum(posterior)
    return posterior


def create_image(simulation, position, dim=(5, 5), uncertainty=False):
    state = simulation.dense_state()
    r0, c0 = xy_to_rc(state.shape[0], position)
    image = np.zeros(dim).astype(np.int8)
    observation = dict()

    half_row = (dim[0]-1)//2
    half_col = (dim[1]-1)//2

    for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
        for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
            r = r0 + dr
            c = c0 + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                if not uncertainty:
                    image[ri, ci] = state[r, c]
                else:
                    element = simulation.group[(r, c)]
                    probs = [observation_probability(element, element.state, o) for o in element.state_space]
                    obs = np.random.choice(element.state_space, p=probs)
                    observation[(r, c)] = obs
                    image[ri, ci] = obs

    return image, observation, rc_to_xy(state.shape[0], (r0+half_row, c0-half_row))


def col_to_x(col):
    return 1*col


def row_to_y(y_limit, row):
    return 1*(y_limit-row-1)


def rc_to_xy(y_limit, rc):
    return col_to_x(rc[1]), row_to_y(y_limit, rc[0])


def x_to_col(x):
    return np.rint((1/1)*x).astype(np.int8)


def y_to_row(y_limit, y):
    return np.rint(y_limit-1-1*y).astype(np.int8)


def xy_to_rc(y_limit, xy):
    return y_to_row(y_limit, xy[1]), x_to_col(xy[0])


def PlotForest(state, axis):
    # fig = pyplot.figure()
    # ax = fig.add_subplot(111, aspect='equal')
    # pyplot.xlim([0, grid_size/2])
    # pyplot.ylim([0, grid_size/2])
    # pyplot.tick_params(axis='both', which='both',
    #                   labelbottom=False, labelleft=False,
    #                   bottom=False, left=False)

    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            x = col_to_x(c)
            y = row_to_y(state.shape[0], r)

            rec = patches.Rectangle((x, y), (1/1), (1/1), alpha=0.6)
            if state[r, c] == 0:
                rec.set_color('green')
            elif state[r, c] == 1:
                rec.set_color('red')
            elif state[r, c] == 2:
                rec.set_color('black')
            else:
                rec.set_color('gray')

            axis.add_patch(rec)

    return axis


def actions2trajectory(position, actions):
    """
    helper function to map a set of actions to a trajectory.

    action convention:
    1 - upper left, 2 - up,   3 - upper right
    4 - left,       0 - stop, 5 - right
    6 - lower left, 7 - down, 8 -lower right

    Inputs:
    - position: numpy array representing (x, y) position
    - actions: list of integers (see convention) describing movement actions

    Returns:
    - trajectory: list of (x, y) positions created by taking actions, where the
                  first element is the input position
    """

    trajectory = []
    x, y = position
    trajectory.append((x, y))
    for a in actions:
        x, y = trajectory[-1]
        if a == 0:
            trajectory.append((x, y))
        elif a == 1:
            trajectory.append((x-1, y+1))
        elif a == 2:
            trajectory.append((x, y+1))
        elif a == 3:
            trajectory.append((x+1, y+1))
        elif a == 4:
            trajectory.append((x-1, y))
        elif a == 5:
            trajectory.append((x+1, y))
        elif a == 6:
            trajectory.append((x-1, y-1))
        elif a == 7:
            trajectory.append((x, y-1))
        elif a == 8:
            trajectory.append((x+1, y-1))

    return trajectory


if __name__ == '__main__':
    # agent_position = np.array([10.25, 9.25])
    agent_position = np.array([21.5, 3.5])
    agent_time = 0

    dimension = 25
    urban_width = 10
    sim = UrbanForest(dimension, urban_width)
    control = defaultdict(lambda: (0, 0))

    belief = dict()
    for key in sim.group.keys():
        element = sim.group[key]
        belief[key] = np.ones(len(element.state_space)) / len(element.state_space)
        # belief[key][element.state] = 1
    agent_filter = ApproxFilter(belief)

    folder = 'sim_images/single_agent_2/'

    for agent_iteration in range(20):
        img, obs, _ = create_image(sim, agent_position, uncertainty=True)

        fig = pyplot.figure(1)
        ax1 = fig.add_subplot(111, aspect='equal', adjustable='box')
        ax1.set_xlim(0, 25)
        ax1.set_ylim(0, 25)

        # plot forest and agent position
        ax1 = PlotForest(sim.dense_state(), ax1)
        ax1.plot(agent_position[0], agent_position[1], linestyle='', Marker='.', MarkerSize=2, color='blue')

        update = False
        if (agent_iteration+1) % 5 == 0:
            update = True
            sim.update()

        def element_update(key, element, prior):
            return filter_element_update(key, element, prior, update, control, obs)
        belief, _ = agent_filter.filter(sim, element_update)

        entropy = {key: ss.entropy(belief[key]) for key in belief.keys()}
        entropy_matrix = np.zeros(sim.dense_state().shape)
        for key in belief.keys():
            entropy_matrix[key[0], key[1]] = entropy[key]

        weights = np.ones_like(img)
        entropy_map = sn.filters.convolve(entropy_matrix, weights, mode='constant', cval=0)

        scores = np.zeros(9)
        for a in range(9):
            new_position = actions2trajectory(agent_position, [a])[-1]
            new_position_rc = xy_to_rc(sim.dims[1], new_position)
            scores[a] = entropy_map[new_position_rc]

        best_action = np.argmax(scores)

        agent_position = actions2trajectory(agent_position, [best_action])[-1]

        filename = folder + 'iteration' + str(agent_iteration+1).zfill(3) + '.png'
        pyplot.savefig(filename, bbox_inches='tight', dpi=300)

        # pyplot.show()
        pyplot.close(fig)

    print()
