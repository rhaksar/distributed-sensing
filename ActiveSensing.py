from FireSimulator import FireSimulator

import matplotlib.pyplot as pyplot
import numpy as np
import scipy.sparse as spp


def predict(belief, neighbor_map):
    belief_predict = np.zeros_like(belief)

    for i in range(belief_predict.shape[0]):
        for j in range(belief_predict.shape[1]):

            b_predict_healthy = 0
            b_predict_fire = 0
            b_predict_burnt = 0

            neighbors = neighbor_map[(i, j)]

            num_neighbors = len(neighbors)
            for jj in range(2 ** num_neighbors):
                xj = np.base_repr(jj, base=2).zfill(num_neighbors)
                fi = xj.count('1')

                belief_neighbors = 1
                for cnt in range(num_neighbors):
                    p = None
                    if int(xj[cnt]) == 0:
                        p = belief[neighbors[cnt][0], neighbors[cnt][1], 0] + \
                            belief[neighbors[cnt][0], neighbors[cnt][1], 2]
                    elif int(xj[cnt]) == 1:
                        p = belief[neighbors[cnt][0], neighbors[cnt][1], 1]

                    belief_neighbors *= p

                b_predict_healthy += (alpha**fi)*belief_neighbors*belief[i, j, 0]
                b_predict_fire += (1-alpha**fi)*belief_neighbors*belief[i, j, 0]

            b_predict_fire += beta*belief[i, j, 1]
            b_predict_burnt += (1-beta)*belief[i, j, 1]
            b_predict_burnt += 1*belief[i, j, 2]

            belief_predict[i, j, 0] = b_predict_healthy
            belief_predict[i, j, 1] = b_predict_fire
            belief_predict[i, j, 2] = b_predict_burnt

    normalize = np.sum(belief_predict, axis=2)[:, :, np.newaxis]
    belief_predict /= normalize

    return belief_predict


if __name__ == "__main__":

    grid_size = 50
    alpha = 0.2763
    beta = np.exp(-0.1)

    nn = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbor_map = {}
    for i in range(grid_size):
        for j in range(grid_size):
            neighbor_idx = []
            for (di, dj) in nn:
                ii = i + di
                jj = j + dj
                if 0 <= ii < grid_size and 0 <= jj < grid_size:
                    neighbor_idx.append((ii, jj))

            neighbor_map[(i, j)] = neighbor_idx

    x = np.linspace(0, 25, 51, endpoint=True)
    y = np.linspace(0, 25, 51, endpoint=True)
    X, Y = np.meshgrid(x, y)

    belief = np.zeros((grid_size, grid_size, 3))

    belief[:, :, :] = np.array([0.9, 0.1, 0])

    iu = np.triu_indices(grid_size, k=32)
    belief[iu[0], iu[1], :] = np.array([0.5, 0.5, 0.0])
    belief = np.flipud(belief)

    # belief[49, 49, :] = np.array([0.25, 0.5, 0.25]) # np.array([0.25, 0.5, 0.25])

    for _ in range(5):
        belief = predict(belief, neighbor_map)

    p_fire = belief[:, :, 1]
    p_fire[p_fire <= 1e-10] = 1e-15
    p_fire[p_fire >= 1-1e-10] = 1-1e-15
    entropy = p_fire*np.log(p_fire) + (1-p_fire)*np.log(1-p_fire)

    pyplot.figure()
    pyplot.pcolor(X, Y, p_fire, cmap=pyplot.cm.Greys, vmin=0, vmax=1)
    pyplot.colorbar()
    pyplot.axis('square')
    pyplot.title('fire probability')

    pyplot.figure()
    pyplot.pcolor(X, Y, entropy, cmap=pyplot.cm.Greys_r, vmin=np.log(0.5), vmax=0)
    pyplot.colorbar()
    pyplot.axis('square')
    pyplot.title('entropy')

    pyplot.show()

    print('done')
