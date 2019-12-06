import matplotlib.patches as patches
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pickle
import scipy.ndimage as sn
import scipy.stats as ss
import sys
import time

from utilities import rc_to_xy
from scheduling import compute_entropy, compute_conditional_entropy, update_information, graph_search
from filter import merge_beliefs, update_belief

base_path = os.path.dirname(os.getcwd())
sys.path.insert(0, base_path + '/simulators')
from fires.LatticeForest import LatticeForest


def schedule_next_meeting_squence():
    pyplot.rc('text', usetex=True)
    pyplot.rc('font', family='serif')
    
    folder = 'sim_images/meetings/figures/'
    filename = folder + 'meetings-03-Sep-2019-1618.pkl'
    with open(filename, 'rb') as handle:
        save_data = pickle.load(handle)

    settings = save_data['settings']
    schedule = save_data['schedule']
    time_series = save_data['time_series']

    np.random.seed(settings.seed)
    sim = LatticeForest(settings.dimension, rng=settings.seed)

    # cmap = 'Greys_r'
    cmap = 'viridis'
    xmin, xmax = 0.5, 17.5
    ymin, ymax = 3.5, 20.5
    # xmin, xmax = 0, 24.5
    # ymin, ymax = 0, 24.5
    ms = 6
    vmin, vmax = 0, 2.55

    node_row = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_col = np.linspace(0, settings.dimension - 1, settings.dimension)
    node_row, node_col = np.meshgrid(node_row, node_col)
    node_locations = np.stack([node_row.T, node_col.T], axis=2)

    t = 4*settings.meeting_interval
    team = time_series[t]['team']
    sub_team_labels = [3, 4]
    sub_team = [team[k] for k in sub_team_labels]

    merged_belief = merge_beliefs(sub_team)

    predicted_belief = merged_belief
    belief_updates = settings.meeting_interval // settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
    conditional_entropy += 0.1

    # figure of weights, agent locations, and agent last meeting locations
    print(np.max(conditional_entropy))
    figure = pyplot.figure()
    axis = figure.add_subplot(111, aspect='equal', adjustable='box')
    axis.imshow(np.flipud(7*conditional_entropy), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    # axis.grid()
    axis.set_xticks([])
    axis.set_yticks([])
    for idx, agent in enumerate(sub_team):
        color = 'C3' if idx == 0 else 'C1'
        x, y = rc_to_xy(settings.dimension, agent.position)  # + settings.cell_side_length
        x += 0.2*np.cos(idx*np.pi)
        y += 0.2*np.sin(idx*np.pi)
        axis.plot(x, y, linestyle='', Marker='o', MarkerSize=ms, Color=color)

        x, y = rc_to_xy(settings.dimension, agent.last)
        axis.plot(x, y, linestyle='', Marker='^', MarkerSize=ms+2, Color=color)
    pyplot.savefig(folder + 'schedule_next_meeting_1.pdf', dpi=300, bbox_inches='tight')

    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, settings)

        # for location in [agent.first, agent.last]:
        #     conditional_entropy = update_information(conditional_entropy, location, settings)
        conditional_entropy = update_information(conditional_entropy, agent.first, settings)

    weights = sn.filters.convolve(conditional_entropy, np.ones(settings.image_size), mode='constant', cval=0)

    # figure of modified weights, agent locations, and agent meeting locations
    print(np.max(weights))
    figure = pyplot.figure()
    axis = figure.add_subplot(111, aspect='equal', adjustable='box')
    axis.imshow(np.flipud(weights), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    # axis.grid()
    axis.set_xticks([])
    axis.set_yticks([])
    for idx, agent in enumerate(sub_team):
        color = 'C3' if idx == 0 else 'C1'
        x, y = rc_to_xy(settings.dimension, agent.position)  # + settings.cell_side_length
        x += 0.2*np.cos(idx*np.pi)
        y += 0.2*np.sin(idx*np.pi)
        axis.plot(x, y, linestyle='', Marker='o', MarkerSize=ms, Color=color)

        x, y = rc_to_xy(settings.dimension, agent.last)
        axis.plot(x, y, linestyle='', Marker='^', MarkerSize=ms+2, Color=color)
    pyplot.savefig(folder + 'schedule_next_meeting_2.pdf', dpi=300, bbox_inches='tight')

    if all([agent.first == agent.last for agent in sub_team]):
        distances = np.maximum.reduce([np.linalg.norm(node_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team])
    else:
        distances = np.maximum.reduce([np.linalg.norm(node_locations - agent.last, ord=np.inf, axis=2)
                                       for agent in sub_team if agent.first != agent.last])
    locations_r, locations_c = np.where(distances == settings.meeting_interval)
    locations = list(zip(locations_r, locations_c))

    options = []
    highest_weight = -1

    for end in locations:
        v = 0

        for agent in sub_team:
            if agent.first == agent.last:
                _, w = graph_search(agent.last, end, 2*settings.meeting_interval, weights, settings)
            else:
                _, w = graph_search(agent.last, end, settings.meeting_interval, weights, settings)

            v += w
        v /= len(sub_team)

        if v > highest_weight:
            highest_weight = v
        options.append((v, end))

    weights[weights >= 0] = np.nan
    for o in options:
        weights[o[1][0], o[1][1]] = 2*o[0]/highest_weight
    options = [end[1] for end in options if end[0] >= 0.9*highest_weight]
    # np.random.shuffle(options)
    meeting = options[0]

    print(np.amax(weights))
    figure = pyplot.figure()
    axis = figure.add_subplot(111, aspect='equal', adjustable='box')
    i = axis.imshow(np.flipud(weights), cmap=cmap, vmin=vmin, vmax=vmax)
    # divider = make_axes_locatable(axis)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(i, shrink=0.95, pad=0.04)
    # figure.colorbar(i, cax=cax)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    # axis.grid()
    axis.set_xticks([])
    axis.set_yticks([])
    for idx, agent in enumerate(sub_team):
        color = 'C3' if idx == 0 else 'C1'
        x, y = rc_to_xy(settings.dimension, agent.position)  # + settings.cell_side_length
        x += 0.2*np.cos(idx*np.pi)
        y += 0.2*np.sin(idx*np.pi)
        axis.plot(x, y, linestyle='', Marker='o', MarkerSize=ms, Color=color)

        x, y = rc_to_xy(settings.dimension, agent.last)
        axis.plot(x, y, linestyle='', Marker='^', MarkerSize=ms+2, Color=color)

    x, y = rc_to_xy(settings.dimension, meeting)
    axis.plot(x, y, linestyle='', Marker='X', MarkerSize=ms+2, color='white')
    pyplot.savefig(folder + 'schedule_next_meeting_3.pdf', dpi=300, bbox_inches='tight')

    # meeting = (9, 13)
    for agent in sub_team:
        if agent.first == agent.last:
            agent.first = meeting
            agent.last = meeting
            agent.budget = 2 * settings.meeting_interval
        else:
            agent.first = agent.last
            agent.last = meeting
            agent.budget = settings.meeting_interval

    conditional_entropy = compute_conditional_entropy(sub_team[0].belief, sim.group, settings)
    conditional_entropy += 0.1

    for agent in sub_team:
        for other_label in agent.other_plans.keys():
            if not agent.other_plans[other_label]:
                continue
            for location in agent.other_plans[other_label]:
                conditional_entropy = update_information(conditional_entropy, location, settings)

        # for location in [agent.first, agent.last]:
        #     conditional_entropy = update_information(conditional_entropy, location, config)

    weights_plot = sn.filters.convolve(conditional_entropy, np.ones(settings.image_size), mode='constant', cval=0)
    print(np.max(weights_plot))

    # figure of weights, agent locations, and agent last meeting locations
    print(np.max(conditional_entropy))
    figure = pyplot.figure()
    axis = figure.add_subplot(111, aspect='equal', adjustable='box')
    # axis.imshow(np.flipud(7*conditional_entropy), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.imshow(np.flipud(weights_plot), cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    # axis.grid()
    axis.set_xticks([])
    axis.set_yticks([])
    for idx, agent in enumerate(sub_team):
        color = 'C3' if idx == 0 else 'C1'
        x, y = rc_to_xy(settings.dimension, agent.position)  # + settings.cell_side_length
        x += 0.2*np.cos(idx*np.pi)
        y += 0.2*np.sin(idx*np.pi)
        axis.plot(x, y, linestyle='', Marker='o', MarkerSize=ms, Color=color)

        x, y = rc_to_xy(settings.dimension, agent.last)
        axis.plot(x, y, linestyle='', Marker='X', MarkerSize=ms+2, Color='white')

        if agent.first == agent.last:
            continue

        x, y = rc_to_xy(settings.dimension, agent.first)
        axis.plot(x, y, linestyle='', Marker='^', MarkerSize=ms + 2, Color=color)

    pyplot.savefig(folder + 'joint_path_planning_1.pdf', dpi=300, bbox_inches='tight')

    plans = dict()
    for iteration, agent in enumerate(sub_team):
        weights = sn.filters.convolve(conditional_entropy, np.ones(settings.image_size), mode='constant', cval=0)
        plans[agent.label] = []

        if agent.first == agent.last:
            # came_from, _ = graph_search(agent.position, agent.last, 2*config.meeting_interval, weights, config)
            # sub_path = get_path(agent.position, agent.last, came_from)
            sub_path = graph_search(agent.position, agent.last, 2*settings.meeting_interval, weights, settings)[0]
        else:
            sub_path = []
            # came_from, _ = graph_search(agent.position, agent.first, config.meeting_interval, weights, config)
            # sub_path.extend(get_path(agent.position, agent.first, came_from))
            # came_from, _ = graph_search(agent.first, agent.last, config.meeting_interval, weights, config)
            # sub_path.extend(get_path(agent.first, agent.last, came_from))

            sub_path.extend(graph_search(agent.position, agent.first, settings.meeting_interval, weights, settings)[0])
            sub_path.extend(graph_search(agent.first, agent.last, settings.meeting_interval, weights, settings)[0])

        for location in sub_path:
            conditional_entropy = update_information(conditional_entropy, location, settings)

        plans[agent.label].extend(sub_path)

        figure = pyplot.figure()
        axis = figure.add_subplot(111, aspect='equal', adjustable='box')
        i = axis.imshow(np.flipud(weights_plot), cmap=cmap, vmin=vmin, vmax=vmax)
        if iteration+2 == 3:
            figure.colorbar(i, shrink=0.95, pad=0.04)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        # axis.grid()
        axis.set_xticks([])
        axis.set_yticks([])
        for idx, agent in enumerate(sub_team):
            color = 'C3' if idx == 0 else 'C1'
            x, y = rc_to_xy(settings.dimension, agent.position)  # + settings.cell_side_length
            x += 0.2*np.cos(idx*np.pi)
            y += 0.2*np.sin(idx*np.pi)
            axis.plot(x, y, linestyle='', Marker='o', MarkerSize=ms, Color=color)

            x, y = rc_to_xy(settings.dimension, agent.last)
            axis.plot(x, y, linestyle='', Marker='X', MarkerSize=ms+2, Color='white', zorder=2)

            if agent.first == agent.last:
                continue
            x, y = rc_to_xy(settings.dimension, agent.first)
            axis.plot(x, y, linestyle='', Marker='^', MarkerSize=ms+2, Color=color)

        for idx, label in enumerate(plans.keys()):
            color = 'C3' if idx == 0 else 'C1'
            for i in range(len(plans[label])-1):
                x1, y1 = rc_to_xy(settings.dimension, plans[label][i])
                if i == 0:
                    x1 += 0.2*np.cos(idx*np.pi)
                    y1 += 0.2*np.sin(idx*np.pi)
                x2, y2 = rc_to_xy(settings.dimension, plans[label][i+1])
                if i < settings.meeting_interval:
                    axis.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2, color=color, zorder=1)
                else:
                    axis.plot([x1, x2], [y1, y2], linestyle=':', linewidth=2, color=color, zorder=1)

        pyplot.savefig(folder + 'joint_path_planning_' + str(iteration+2) + '.pdf', dpi=300, bbox_inches='tight')

    # pyplot.show()


if __name__ == '__main__':
    schedule_next_meeting_squence()
