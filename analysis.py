import matplotlib.pyplot as pyplot
import numpy as np
import pickle


def get_data(files):
    data = list()
    for fn in files:
        filename = 'Benchmark/' + fn
        with open(filename, 'rb') as handle:
            save_data = pickle.load(handle)

        time_data = [np.median([np.mean(save_data[key][key2]) for key2 in save_data[key].keys()]) for key in
                     save_data.keys()]
        data.append(np.median(time_data))
    return data


def get_data2(files):
    data = dict()
    for idx, fn in enumerate(files):
        data[idx] = []
        filename = 'Benchmark/' + fn
        with open(filename, 'rb') as handle:
            save_data = pickle.load(handle)

        for seed in save_data.keys():
            T = len(save_data[seed].keys()) - 1
            data[idx].append(np.mean(save_data[seed][T]))

    return data


if __name__ == '__main__':
    # filename = 'Benchmark/benchmark-tau04C02pc0.75.pkl'
    # filename = 'Benchmark/benchmark-tau04C05pc0.75.pkl'
    # filename = 'Benchmark/benchmark-tau08C10pc0.95.pkl'
    # with open(filename, 'rb') as handle:
    #     save_data = pickle.load(handle)
    #
    # for sim in save_data.keys():
    #     time_data = [np.mean(save_data[sim][t]) for t in save_data[sim].keys()]
    #     pyplot.plot(time_data)
    #
    # pyplot.show()

    # data = list()
    # for key in save_data.keys():
    #     time_data = list()
    #     time_data.append(np.median([np.median(save_data[key][key2]) for key2 in save_data[key].keys()]))
    #     data.extend(time_data)
    #
    # pyplot.boxplot(data)
    # pyplot.show()

    files1 = ['benchmark-tau04C03pc0.95.pkl', 'benchmark-tau04C05pc0.95.pkl', 'benchmark-tau04C10pc0.95.pkl']
    files2 = ['benchmark-tau08C03pc0.95.pkl', 'benchmark-tau08C05pc0.95.pkl', 'benchmark-tau08C10pc0.95.pkl']
    files3 = ['benchmark-tau12C03pc0.95.pkl', 'benchmark-tau12C05pc0.95.pkl', 'benchmark-tau12C10pc0.95.pkl']

    data1 = get_data2(files1)
    data2 = get_data2(files2)
    data3 = get_data2(files3)

    pyplot.rc('text', usetex=True)
    pyplot.rc('font', family='serif')
    figure = pyplot.figure()
    ax = figure.add_subplot('111')
    ax.grid()
    x = [3, 5, 10]
    taus = [4, 8, 12]
    for idx, data in enumerate([data1, data2, data3]):
        y = [np.median(data[key]) for key in data.keys()]
        yerr = list()
        yerr.append([np.median(data[key]) - np.percentile(data[key], 25) for key in data.keys()])
        yerr.append([np.percentile(data[key], 75) - np.median(data[key]) for key in data.keys()])
        ax.errorbar(x, y, yerr=yerr, Marker='x', MarkerSize=6, capsize=3, label=r'$\tau=' + str(taus[idx]) + '$')
    ax.set_xlabel('Number of Agents', fontsize=16)
    ax.set_ylabel('Coverage Metric', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)
    # for data in [data1, data2, data3]:
    #     ax.plot([3, 5, 10], [np.median(data[key]) for key in data.keys()], Marker='x', LineStyle='-')
    # ax.boxplot(data1.values(), showfliers=False, widths=0)
    # ax.boxplot(data2.values(), showfliers=False, widths=0)
    # ax.boxplot(data3.values(), showfliers=False, widths=0)

    pyplot.savefig('performance.pdf', dpi=300, bbox_inches='tight')

    # pyplot.show()

    # figure = pyplot.figure()
    # ax = figure.add_subplot(111)
    # ax.plot([3, 5, 10], data1, Marker='x')
    # ax.plot([3, 5, 10], data2, Marker='x')
    # # ax.set_ylim(0.80, 1.0)
    #
    # figure = pyplot.figure()
    # ax = figure.add_subplot(111)
    # ax.plot([3, 5, 10], data1, Marker='x')
    # ax.plot([3, 5, 10], data3, Marker='x')
    # # ax.set_ylim(0.80, 1.0)
    #
    # pyplot.show()
    # print()

    # filename = 'sim_images/meetings/meetings-04-Sep-2019-1923.pkl'
    # with open(filename, 'rb') as handle:
    #     save_data = pickle.load(handle)
    #
    # settings = save_data['settings']
    # schedule = save_data['schedule']
    # time_series = save_data['time_series']
    # print('number of robots:', settings.team_size)
    #
    # entropy_time_history = np.zeros((settings.team_size, len(time_series.keys())))
    # accuracy_time_history = np.zeros((settings.team_size, len(time_series.keys())))
    #
    # for t in range(len(time_series.keys())):
    #     for label in time_series[t]['entropy'].keys():
    #         entropy_time_history[label-1, t] = np.mean(time_series[t]['entropy'][label])
    #         accuracy_time_history[label-1, t] = time_series[t]['accuracy'][label]
    #
    # figure = pyplot.figure()
    # ax = figure.add_subplot(111)
    # ax.set_ylim(0, 1)
    # # ax.set_xlim(0, 21)
    #
    # # for label in range(1, settings.team_size):
    # #     ax.plot(entropy_time_history[label-1, :])
    #
    # for label in range(1, settings.team_size):
    #     ax.plot(accuracy_time_history[label-1, :])
    #
    # pyplot.show()
    #
    # print('mean accuracy per robot per time step: {0:0.2f}'.format(np.mean(accuracy_time_history)))
