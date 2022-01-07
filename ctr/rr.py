from . lr import LR
from bidding.bid_strategy import BidStrategy
from base.dataset import Dataset
import math
import random

class RR(LR):
    def __init__(self, train_data, test_data):
        LR.__init__(self, train_data, test_data)

    def init_bid_strategy(self):
        self.bid_strategy = BidStrategy(self.camp_v, self.mu)

    def train(self):
        random.seed(10)
        train_data = self.train_data
        iter_id = train_data.init_index()
        while not train_data.reached_tail(iter_id):
            data = train_data.get_next_data(iter_id)
            y = data[0]
            feature = data[2:len(data)]
            ctr = super().predict(self.weight, feature, train_flag=True)
            phi = 1.0 / (1.0 + self.mu)
            bp = self.bid_strategy.bidding_opt(ctr)
            pz = self.train_data.landscape.get_probability(bp)
            #scale_x = (phi * ctr - y) * phi * math.pow(self.camp_v, 2) * pz * self.scale
            scale_x = (phi * ctr - y) * phi * self.camp_v * pz * self.scale
            for idx in feature:
                self.weight[idx] = self.weight[idx] * self.reg_update_param - self.alpha * scale_x
