import matplotlib.pyplot as pyplot
import numpy as np
import scipy.cluster as spc
import scipy.spatial as sps

if __name__ == "__main__":

    num_pos = 10
    max_distance = 4

    positions = np.random.uniform(0.25, 24.75, (num_pos, 2))
    # positions = np.random.uniform(0, 5, (num_pos, 2))

    # positions = np.zeros((10, 2))
    # positions[:, 1] = 12
    # positions[:, 0] = np.linspace(1, 24, 10, endpoint=True)

    # pyplot.plot(positions[:, 0], positions[:, 1], linestyle='', Marker='.', MarkerSize=10)
    # pyplot.xlim([0, 25])
    # pyplot.ylim([0, 25])

    # dist_matrix = sps.distance.cdist(positions, positions)
    # test = np.zeros_like(dist_matrix).astype(np.int8)
    # test[np.where(dist_matrix < max_distance)] = 1

    Z = spc.hierarchy.linkage(positions, method='ward')
    clusters = spc.hierarchy.fcluster(Z, max_distance, criterion='distance')

    # print(np.around(positions, decimals=2))
    # print(np.around(dist_matrix, decimals=2))
    # print(Z)
    print(clusters)

    fig = pyplot.figure()

    ax1 = fig.add_subplot(121)
    spc.hierarchy.dendrogram(Z, no_plot=False, ax=ax1)
    ax1.axhline(y=max_distance, color='k')
    ax1.set_xlabel('agent number')
    ax1.set_ylabel('metric value')

    ax2 = fig.add_subplot(122)
    ax2.set_xlim([0, 25])
    ax2.set_ylim([0, 25])
    ax2.scatter(positions[:, 0], positions[:, 1], c=clusters, cmap='prism', edgecolor='black')
    # ax2.set_xlabel('x position')
    # ax2.set_ylabel('y position')
    for i in range(num_pos):
        ax2.text(positions[i, 0], positions[i, 1], str(clusters[i]), color='black', fontsize=12)

    # pyplot.savefig('clustering.png', dpi=300, bbox_inches='tight')

    pyplot.show()
