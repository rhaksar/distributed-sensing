import numpy as np


class UAV(object):

    def __init__(self, label=None, position=None, belief=None, image_size=None):
        self.label = label
        # self.deployed = False
        # self.time = 1

        self.position = position
        # self.next_position = None

        self.image_size = image_size
        self.belief = belief
        # self.MImetric = None

        self.budget = None
        self.first = None
        self.last = None

        # self.meetings = []
        self.plan = []
        self.other_plans = dict()

    # def deploy(self, time, config):
    #     if self.deployed:
    #         return
    #
    #     # check if correct time to be deployed
    #     if (time-1)%config.deploy_interval==0 and self.id <= 2*((time-1)/config.deploy_interval + 1):
    #         self.position = config.deploy_locations[1] if self.id%2==0 else config.deploy_locations[0]
    #         self.deployed = True
    #
    #     return
