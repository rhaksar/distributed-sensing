import cvxpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as pyplot
import numpy as np
from scipy.stats import multivariate_normal

if __name__ == "__main__":
    x = np.linspace(0, 4, 100, endpoint=True)
    y = np.linspace(0, 4, 100, endpoint=True)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # mu = np.array([1, 2])
    # cov = np.array([[.5, .25], [.25, .5]])
    # rv = multivariate_normal(mu, cov)
    # Z = rv.pdf(pos)

    # mu = np.array([1, 2])
    mu_list = [np.array([2.7, 0.7]), np.array([3, 3]), np.array([0.5, 3])]
    d = 0.5
    Z = np.zeros_like(X)
    # for mu in mu_list:
        # Z += np.maximum(np.linalg.norm(pos-mu, ord=1, axis=2) - d, 0)
        # Z += np.linalg.norm(pos-mu, ord=1, axis=2)
        # Z += np.exp(np.linalg.norm(pos-mu, ord=np.inf, axis=2))

    Z = np.zeros((X.shape[0], X.shape[1], len(mu_list)))
    for idx, mu in enumerate(mu_list):
        Z[:, :, idx] += np.maximum(np.linalg.norm(pos-mu, ord=np.inf, axis=2) - d, 0)

    Z = np.amin(Z, axis=2)

    # fig = pyplot.figure()
    # ax = fig.add_subplot(121, aspect='equal')
    # ax.pcolor(X, Y, Z)
    # ax = fig.add_subplot(122, aspect='equal', projection='3d')
    # ax.plot_surface(X, Y, Z)

    agent = {'position': np.array([0, 0])}

    d = 0.1
    mu_list = [np.array([2.75, 0.75]), np.array([2.75, 1.25]), np.array([2.75, 1.75]), np.array([2.75, 2.25]),
               np.array([1.25, 2.75]), np.array([0.25, 1.25])]
    mu_list.sort(key=lambda s: np.linalg.norm(s - agent['position'], ord=np.inf))

    fig = pyplot.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for el in mu_list:
        ax.plot(el[0], el[1], Marker='.', Markersize=10, color='red')

    for idx, target in enumerate(mu_list):
        if idx == 0:
            x_0 = agent['position']
        else:
            x_0 = next_x_0

        T = int(np.ceil((1/0.5)*np.linalg.norm(x_0 - target, ord=2)))
        T = np.maximum(T, 2)
        x = cvxpy.Variable((2, T+1))
        u = cvxpy.Variable((2, T))
        w = cvxpy.Variable((2, T))

        states = []
        for t in range(T):
            cost = cvxpy.maximum(cvxpy.norm(x[:, t+1]-target, p='inf') - d, 0) + cvxpy.norm(u[:, t], p=2)
            # cost = cvxpy.norm(x[:, t+1] - mu_list[0], p=2) + cvxpy.norm(w[:, t], p=2)
            constraints = [x[:, t+1] == x[:, t] + u[:, t],
                           cvxpy.norm(u[:, t], p=2) <= 0.5, cvxpy.norm(w[:, t], p=2) <= 0.5,
                           cvxpy.norm(u[:, t], p=2) + cvxpy.norm(w[:, t], p=2) <= 0.5]
            states.append(cvxpy.Problem(cvxpy.Minimize(cost), constraints))

        constraints = [cvxpy.maximum(cvxpy.norm(x[:, T] - target, p='inf') - d, 0) <= 0, x[:, 0] == x_0]
        # constraints = [x[:, T-1] + w[:, T-1] == mu_list[0], x[:, 0] == x_0]
        states.append(cvxpy.Problem(cvxpy.Minimize(1 - cvxpy.sum(w[:, T-1])), constraints))

        problem = cvxpy.sum(states)
        problem.solve()

        next_x_0 = x.value[:, T]

        ax.plot(x.value[0, :], x.value[1, :], Marker='.', Markersize=10, color='black')

        # fig = pyplot.figure()
        # ax_new = fig.add_subplot(211, aspect='equal')
        # ax_new.plot(range(T), np.linalg.norm(u.value, axis=0, ord=2))
        # ax_new = fig.add_subplot(212, aspect='equal')
        # ax_new.plot(range(T), np.linalg.norm(w.value, axis=0, ord=2))

    pyplot.show()