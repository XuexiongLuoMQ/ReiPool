from coarse_selector import CoarseSelector
from threshold_selector import ThresholdSelector


class Agent(object):
    def __init__(self, coarse_net, threshold_net):
        self.coarse_net = coarse_net
        self.threshold_net = threshold_net


    def learn(self):
        pass

    def train(self):
        pass
        