import itertools
from collections import defaultdict
import numpy as np
import warnings


class FireSimulator(object):
    """
    A simulation of a forest fire using a discrete probabilistic grid model.
    The model is adapted from literature:
      A. Somanath, S. Karaman, and K. Youcef-Toumi, "Controlling stochastic
      growth processes on lattices: Wildfire management with robotic fire
      extinguishers," in 53rd IEEE Conference on Decision and Control (CDC),
      2014, pp. 1432-1327
    """

    def __init__(self, dimension, rng=None, fire_init=None,
                 alpha=None, beta=None, model='exponential'):
        """
        Initializes a simulation object.
        Each tree in the grid model has three states:
          0 - healthy tree
          1 - on fire tree
          2 - burnt tree

        Inputs:
        - dimension: size of forest, integer or tuple of integers
                     if an integer, the forest is a square grid of size (dimension, dimension)
        - rng: random number generator seed to use for deterministic sampling
        - fire_init: list of tuples of (r, c) coordinates describing
                     positions of initial fires
        - alpha: fire propagation parameter, as a defaultdict with array indices (r, c) as keys
        - beta: fire persistence parameter, as a defaultdict with array indices (r, c) as keys
        - model: method to convert fire propagation parameter to probability of catching on fire
                 options - 'linear' or 'exponential'
        """

        self.dims = (dimension, dimension) if isinstance(dimension, int) else dimension

        self.alpha = defaultdict(lambda: 1-0.2763) if alpha is None else alpha
        self.beta = defaultdict(lambda: np.exp(-1/10)) if beta is None else beta
        self.model = model

        self.state = np.zeros(self.dims).astype(np.uint8)

        # number of [healthy, on fire, burnt] trees
        self.stats = np.zeros(3).astype(np.uint32)
        self.stats[0] += self.dims[0]*self.dims[1]

        # tree states and neighbor set
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2
        self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if rng is not None:
            np.random.seed(rng)

        # initialize fires at initial time
        self.iter = 0
        self.fires = []
        if fire_init is not None:
            self.fires = fire_init
            self.iter += 1
            for (r, c) in fire_init:
                self.state[r, c] = 1

            self.stats[0] -= len(fire_init)
            self.stats[1] += len(fire_init)
        else:
            self._start_fire()

        self.end = False
        self.early_end = False
        return

    def _start_fire(self):
        """
        Helper function to specify initial fire locations in the forest.
        Assumes forest is at least 4x4 size.

        Inputs/Outputs:
         None
        """
        if self.dims[0] < 4 or self.dims[1] < 4:
            raise ValueError('Fire initialization requires a forest size of at least 4x4')

        # start a square of fires at center
        r_center = np.ceil(self.dims[0]/2, dtype=np.uint8)
        c_center = np.ceil(self.dims[1]/2, dtype=np.uint8)
        delta = [k for k in range(-1, 3)]
        delta = itertools.product(delta, delta)

        for (dr, dc) in delta:
            r, c = r_center+dr, c_center+dc
            self.fires.append((r, c))
            self.state[r, c] = 1

        self.stats[0] -= len(self.fires)
        self.stats[1] += len(self.fires)
        self.iter += 1
        return

    def _persists(self, fire, action, dbeta):
        """
        Helper function to determine if a fire should continue to burn. Also
        defines how the action affects the state update.

        Inputs:
        - fire: tuple of (r, c) position
        - action: list of trees that should be treated at current time step
        - dbeta: reduction in fire persistence parameter if tree is treated

        Returns:
          True if fire continues to burn
          False if fire burns out
        """

        delta = dbeta if fire in action else 0
        threshold = self.beta[fire]-delta

        # check for invalid probability
        if threshold > 1:
            warnings.warn('Fire persistence parameter and control effect are set such that '
                          'the probability of persisting is greater than one')
        elif threshold < 0:
            warnings.warn('Fire persistence parameter and control effect are set such that '
                          'the probability of persisting is less than zero')

        sample = np.random.rand()
        if sample > threshold:
            self.state[fire[0], fire[1]] = 2
            self.stats[1] -= 1
            self.stats[2] += 1
            return False
        return True

    def step(self, action, dbeta=0):
        """
        Method to update the state of the forest fire.

        Inputs:
        - action: list of tuples of (r, c) indices describing treatment location
        - dbeta: reduction in fire persistent parameter due to treatment
        """

        if self.end:
            print("fire extinguished")
            return

        # assume that fires are not able to spread to healthy trees
        # flag will be False if there exists a fire with at least one healthy neighbor
        self.early_end = True

        add = []  # list of trees that will catch on fire
        checked = []  # list of healthy trees that have been simulated

        # fire spread step:
        #     iterate over fires, find their healthy neighbors, and sample to
        #     determine if each healthy neighbor catches fire
        for (r, c) in self.fires:

            for (dr, dc) in self.neighbors:
                rn, cn = r + dr, c + dc

                # check for neighbors of fires that are healthy
                # and have not been simulated yet
                if 0 <= rn < self.dims[0] and 0 <= cn < self.dims[1]:
                    if (rn, cn) not in checked and self.state[rn, cn] == self.healthy:
                        self.early_end = False

                        # determine the number of trees on fire surrounding the healthy tree
                        num_neighbor_fires = 0
                        for (dr2, dc2) in self.neighbors:
                            rn2, cn2 = rn + dr2, cn + dc2

                            if 0 <= rn2 < self.dims[0] and 0 <= cn2 < self.dims[1]:
                                if self.state[rn2, cn2] == self.on_fire:
                                    num_neighbor_fires += 1

                        # determine the probability of catching on fire and sample
                        if self.model == 'exponential':
                            p = 1-(self.alpha[(r, c)]**num_neighbor_fires)
                        elif self.model == 'linear':
                            p = self.alpha[(r, c)]*num_neighbor_fires
                        else:
                            raise ValueError('Invalid model name provided')

                        # check for invalid probability
                        if p < 0:
                            warnings.warn('Fire propagation parameter is set such that the '
                                          'probability of catching on fire is less than zero')
                        if p > 1:
                            warnings.warn('Fire propagation parameter is set such that the '
                                          'probability of catching on fire is greater than one')

                        # sample to determine if healthy tree catches on fire
                        if np.random.rand() <= p:
                            add.append((rn, cn))

                        checked.append((rn, cn))

        # fire burn out step, including action
        self.fires = [fire for fire in self.fires if self._persists(fire, action, dbeta)]

        # update list of current fires
        for (r, c) in add:
            self.fires.append((r, c))
            self.state[r, c] = 1

        self.stats[0] -= len(add)
        self.stats[1] += len(add)

        # terminate if no fires
        if not self.fires:
            self.iter += 1
            self.end = True
            return

        self.iter += 1
        return
