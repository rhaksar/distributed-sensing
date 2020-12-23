class UAV(object):
    """
    Aerial vehicle definition.
    """

    def __init__(self, label, belief, image_size):
        # a numeric id
        self.label = label

        self.position = ()

        self.image_size = image_size
        self.belief = belief

        # planning budget and meeting locations
        self.budget = 0
        self.first = ()
        self.last = ()

        # sequence of positions indicating planned trajectory, and plans by other UAVs, to prevent duplicate coverage
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
