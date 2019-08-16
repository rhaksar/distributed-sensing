import numpy as np


class UAV(object):

    def __init__(self, label=None, position=None, belief=None, image_size=None):
        self.label = label
        # self.deployed = False

        self.position = position

        self.image_size = image_size
        self.belief = belief

        self.budget = None
        self.first = None
        self.last = None

        # self.meetings = []
        self.plan = []
        self.other_plans = dict()

    def __repr__(self):
        return 'UAV(\'label\': ' + str(self.label) + ', \'position\': ' + str(self.position) + ', \'first\': ' + \
               str(self.first) + ', \'last\': ' + str(self.last) + ', \'budget\': ' + str(self.budget) + \
               ', \'plan\': ' + str(self.plan) + ', \'other_plans\': ' + str(self.other_plans) + ')'

    def __str__(self):
        return 'UAV ' + str(self.label) + '\n' + 'position: ' + str(self.position) + '\n' + 'first: ' + \
               str(self.first) + '\n' + 'last: ' + str(self.last) + '\n' + 'budget: ' + str(self.budget) + \
               '\n' + 'plan: ' + str(self.plan) + '\n' + 'other plans: ' + str(self.other_plans)

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
