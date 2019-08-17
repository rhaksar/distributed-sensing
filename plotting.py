import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
import numpy as np
import pickle
import scipy.stats as ss

from utilities import rc_to_xy
from scheduling import compute_entropy

if __name__ == '__main__':
    filename = 'sim_images/meetings/meetings-16-Aug-2019-1721.pkl'
    with open(filename, 'rb') as handle:
        save_data = pickle.load(handle)

    settings = save_data['settings']
    schedule = save_data['schedule']
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
    axis_left.set_xticks([])
    axis_left.set_yticks([])
    axis_right = figure.add_subplot(122, aspect='equal', adjustable='box')
    axis_right.set_xticks([])
    axis_right.set_yticks([])

    team = time_series[0]['team']
    agent_plotting = {agent.label: dict() for agent in team.values()}
    for agent in team.values():
        agent_plotting[agent.label]['path'], = axis_left.plot([], [], linestyle='-', linewidth=0.5,
                                                              Color='white', zorder=1)
        agent_plotting[agent.label]['meetings'], = axis_left.plot([], [], linestyle='', Marker='x', MarkerSize=2,
                                                                  Color='white', zorder=1)
        agent_plotting[agent.label]['position'], = axis_left.plot([], [], linestyle='', Marker='o', MarkerSize=2,
                                                                  Color='blue', zorder=2)

    meeting_plotting, = axis_left.plot([], [], linestyle='', marker='o', markersize=1, color='blue', zorder=1)
    next_meetings = 0

    max_entropy_value = ss.entropy((1/3)*np.ones(3))

    for tree in trees:
        axis_left.add_artist(tree)

    for t in range(len(time_series.keys())):
        print('[plotting] time {0:d}'.format(t))
        # axis_left.cla()

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
        agent_label_entropy = 1
        entropy = compute_entropy(team[agent_label_entropy].belief, settings)
        i = axis_right.imshow(entropy, vmin=0, vmax=max_entropy_value,
                              extent=[0, settings.dimension, 0, settings.dimension])
        # figure.colorbar(i, ax=axis_right)

        agent_in_meeting = np.zeros(settings.team_size)
        data_x, data_y = [], []
        if t % settings.meeting_interval == 0:
            for meeting in schedule[next_meetings]:
                angle_increment = 2*np.pi/len(meeting)
                for idx, label in enumerate(meeting):
                    agent_in_meeting[label-1] = 1
                    x, y = np.asarray(rc_to_xy(settings.dimension, team[label].position)) + settings.cell_side_length
                    data_x.append(x + 0.2*np.cos(idx*angle_increment))
                    data_y.append(y + 0.2*np.sin(idx*angle_increment))
            next_meetings = 0 if next_meetings + 1 > len(schedule) - 1 else next_meetings + 1
        meeting_plotting.set_data(data_x, data_y)

        for agent in team.values():
            if agent_in_meeting[agent.label-1] == 1:
                agent_plotting[agent.label]['position'].set_data([], [])
                continue
            x, y = np.asarray(rc_to_xy(settings.dimension, agent.position)) + settings.cell_side_length
            agent_plotting[agent.label]['position'].set_data(x, y)

        # else:
        #     data_x, data_y = [], []
        #     for agent in team.values():
        #         x, y = np.asarray(rc_to_xy(settings.dimension, agent.position)) + settings.cell_side_length
        #         data_x.append(x)
        #         data_y.append(y)
        #     meeting_plotting.set_data(data_x, data_y)
        #     meeting_plotting.set_markersize(2)

        # for agent in team.values():
        #     xy = np.asarray(rc_to_xy(settings.dimension, agent.position)) + settings.cell_side_length
        #     agent_plotting[agent.label]['position'].set_data(xy[0], xy[1])

        for agent in team.values():
            data_x, data_y = [], []
            for element in agent.plan:
                x, y = np.asarray(rc_to_xy(settings.dimension, element)) + settings.cell_side_length
                data_x.append(x)
                data_y.append(y)
            agent_plotting[agent.label]['path'].set_data(data_x, data_y)

            data_x, data_y = [], []
            for element in [agent.first, agent.last]:
                x, y = np.asarray(rc_to_xy(settings.dimension, element)) + settings.cell_side_length
                data_x.append(x)
                data_y.append(y)
                if element == agent.last:
                    continue
            agent_plotting[agent.label]['meetings'].set_data(data_x, data_y)

        axis_left.figure.canvas.draw()
        filename = 'sim_images/meetings/' + 'iteration' + str(t).zfill(3) + '.png'
        pyplot.savefig(filename, bbox_inches='tight', dpi=300)

    final_time = len(time_series.keys())-1
    extra_iterations = 10
    for t in range(extra_iterations):
        print('[plotting] time {0:d} [{1:d}]'.format(final_time, t+1))
        filename = 'sim_images/meetings/' + 'iteration' + str(final_time+1+t).zfill(3) + '.png'
        pyplot.savefig(filename, bbox_inches='tight', dpi=300)

    print('done')
