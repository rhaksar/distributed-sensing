import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import pickle

from utilities import rc_to_xy
from scheduling import compute_entropy

if __name__ == '__main__':
    filename = 'sim_images/meetings/meetings-01-Aug-2019-1850.pkl'
    with open(filename, 'rb') as handle:
        save_data = pickle.load(handle)

    settings = save_data['settings']
    time_series = save_data['time_series']

    tree_patch_map = {}
    trees = []

    for r in range(settings.dimension):
        for c in range(settings.dimension):
            xy = rc_to_xy(settings.dimension, (r, c))
            tree_patch_map[(r, c)] = len(tree_patch_map)
            trees.append(patches.Rectangle(xy, 1, 1, alpha=0.6, zorder=0))

    figure = pyplot.figure(1)
    axis_left = figure.add_subplot(121, aspect='equal', adjustable='box')
    axis_left.set_xlim(0, 25)
    axis_left.set_ylim(0, 25)
    axis_right = figure.add_subplot(122, aspect='equal', adjustable='box')

    team = time_series[0]['team']
    agent_plotting = {agent.label: dict() for agent in team.values()}
    for agent in team.values():
        agent_plotting[agent.label]['position'], = axis_left.plot([], [], linestyle='', Marker='o', MarkerSize=2,
                                                                  Color='blue', zorder=2)

    for tree in trees:
        axis_left.add_artist(tree)

    for t in range(len(time_series.keys())):
        process_state = time_series[t]['process_state']
        for r in range(settings.dimension):
            for c in range(settings.dimension):
                idx = tree_patch_map[(r, c)]
                if process_state[r, c] == 0:
                    trees[idx].set_color('green')
                elif process_state[r, c] == 1:
                    trees[idx].set_color('red')
                elif process_state[r, c] == 2:
                    trees[idx].set_color('black')

        team = time_series[t]['team']
        entropy = compute_entropy(team[1].belief, settings)
        axis_right.imshow(entropy, vmin=0, vmax=2, extent=[0, settings.dimension, 0, settings.dimension])

        for agent in team.values():
            xy = np.asarray(rc_to_xy(settings.dimension, agent.position)) + settings.cell_side_length
            agent_plotting[agent.label]['position'].set_data(xy[0], xy[1])

        axis_left.figure.canvas.draw()
        filename = 'sim_images/meetings/' + 'iteration' + str(t).zfill(3) + '.png'
        pyplot.savefig(filename, bbox_inches='tight', dpi=150)
