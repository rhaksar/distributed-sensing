import matplotlib.pyplot as pyplot
import numpy as np
import pickle


if __name__ == '__main__':
    pyplot.rc('text', usetex=True)
    pyplot.rc('font', family='serif')
    figure = pyplot.figure()
    ax = figure.add_subplot('111')
    ax.grid()
    ax.set_xlabel('Number of Agents', fontsize=16)
    ax.set_ylabel('Fire Coverage (\%)', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    error_alpha = 0.75

    rho = 2
    tau_set = [4, 8, 12]
    robot_set = [2, 5, 10, 20]
    ax.set_xlim([0, 21])
    ax.set_xticks([2, 5, 10, 15, 20])
    directory = '/home/ravi/Desktop/Benchmark/'

    for tau in tau_set:
        benchmark_data = []
        benchmark_error = np.zeros((2, len(robot_set)))

        for idx, robot in enumerate(robot_set):
            filename = directory
            filename += 'benchmark-rho' + str(rho).zfill(2) + 'tau' + str(tau).zfill(2) + 'C' + str(robot).zfill(2) \
                        + 'pc0.95.pkl'
            with open(filename, 'rb') as handle:
                save_data = pickle.load(handle)

            data = [100*np.mean(save_data[seed]['coverage']) for seed in save_data.keys()]

            benchmark_data.append(np.median(data))
            benchmark_error[0, idx] = np.median(data) - np.percentile(data, 25)
            benchmark_error[1, idx] = np.percentile(data, 75) - np.median(data)

        makers, caps, bars = ax.errorbar(robot_set, benchmark_data, yerr=benchmark_error, Marker='.', Markersize=6, capsize=3,
                                         label=r'$\tau = ' + str(tau) + '$')
        [bar.set_alpha(error_alpha) for bar in bars]
        [cap.set_alpha(error_alpha) for cap in caps]

    tau = 8
    baseline_ncomm_data = []
    baseline_ncomm_error = np.zeros((2, len(robot_set)))
    for idx, robot in enumerate(robot_set):
        filename = directory
        filename += 'baseline-ncomm-rho' + str(rho).zfill(2) + 'tau' + str(tau).zfill(2) + 'C' + \
                    str(robot).zfill(2) + 'pc0.95.pkl'
        with open(filename, 'rb') as handle:
            save_data = pickle.load(handle)

        data = [100*np.mean(save_data[seed]['coverage']) for seed in save_data.keys()]

        baseline_ncomm_data.append(np.median(data))
        baseline_ncomm_error[0, idx] = np.median(data) - np.percentile(data, 25)
        baseline_ncomm_error[1, idx] = np.percentile(data, 75) - np.median(data)

    makers, caps, bars = ax.errorbar(robot_set, baseline_ncomm_data, yerr=baseline_ncomm_error,
                                     Marker='.', Markersize=6, capsize=3,
                                     label=r'no comm.')
    [bar.set_alpha(error_alpha) for bar in bars]
    [cap.set_alpha(error_alpha) for cap in caps]

    baseline_ycomm_data = []
    baseline_ycomm_error = np.zeros((2, len(robot_set)))
    for idx, robot in enumerate(robot_set):
        filename = directory
        filename += 'baseline-ycomm-rho' + str(rho).zfill(2) + 'tau' + str(tau).zfill(2) + 'C' + \
                    str(robot).zfill(2) + 'pc0.95.pkl'
        with open(filename, 'rb') as handle:
            save_data = pickle.load(handle)

        data = [100*np.mean(save_data[seed]['coverage']) for seed in save_data.keys()]

        baseline_ycomm_data.append(np.median(data))
        baseline_ycomm_error[0, idx] = np.median(data) - np.percentile(data, 25)
        baseline_ycomm_error[1, idx] = np.percentile(data, 75) - np.median(data)

    makers, caps, bars = ax.errorbar(robot_set, baseline_ycomm_data, yerr=baseline_ycomm_error,
                                     Marker='.', Markersize=6, capsize=3,
                                     label=r'team comm.')
    [bar.set_alpha(error_alpha) for bar in bars]
    [cap.set_alpha(error_alpha) for cap in caps]

    ax.legend(fontsize=14, ncol=2)

    pyplot.savefig('performance_' + str(rho).zfill(2) + '.pdf', dpi=300, bbox_inches='tight')
    pyplot.show()

