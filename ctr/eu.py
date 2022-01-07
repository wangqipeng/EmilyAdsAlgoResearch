from . lr import LR
from market.bid_landscape import BidLandscape
from bidding.opt_bid import OptBid
from base.dataset import Dataset
import math
import random
import base.tool as tool
from base.tool import logger

class EU(LR):
    def __init__(self, config, train, test):
        LR.__init__(self, config, train, test)
        
    def init_parameters(self):
        self.camp_v = self.train_data.get_statistics()['ecpc']
        if self.ds_ratio > 0:
            self.ori_camp_v = self.train_data.get_statistics()['ori_ecpc']
        self.budget = int(self.test_data.get_statistics()['cost_sum'] / self.budget_prop)
        self.mu = 0.0
    
    def init_bid_strategy(self):
        self.bid_strategy = OptBid(self.camp_v, self.mu)

    def init_market_strategy(self, mkt_model):
        self.mkt_model = mkt_model
        self.use_market = True

    def init_bid_landscape(self, landscape):
        self.landscape = landscape

    def get_bid_strategy(self):
        if self.bid_strategy == None:
            logger.error("ERROR: Please init bid strategy first.")
        return self.bid_strategy

    def train(self):
        random.seed(10)
        train_data = self.train_data
        iter_id = train_data.init_index()
        while not train_data.reached_tail(iter_id):
            data = train_data.get_next_data(iter_id)
            y = data[0]
            feature = data[2:len(data)]
            ctr = self.estimate_ctr(self.weight, feature, train_flag=True, ctr_avg=train_data.get_statistics()['ctr'])
            phi = 1.0 / (1.0 + self.mu)
            bp = self.bid_strategy.bid(ctr)
            if self.use_market:
                pz = self.mkt_model.get_win_probability(bp, feature)
            else:
                pz = self.landscape.get_probability(bp)
            grad = (phi * ctr - y) * phi * math.pow(self.camp_v, 2) * pz * ctr * (1 - ctr) * self.eu_scale
            #scale_x = math.pow(self.camp_v, 2) * (ctr - y) *  pz * ctr * (1 - ctr) * config.eu_scale
            for idx in feature:
                self.weight[idx] = self.weight[idx] * self.reg_update_param - self.lr_alpha * grad
                #self.weight[idx] = self.weight[idx] - self.lr_alpha * grad

    def estimate_ctr(self, weight, feature, train_flag = False, ctr_avg=0.125):
        value = 0.0
        for idx in feature:
            if idx in weight:
                value += weight[idx]
            elif train_flag:
                if idx == 0:
                    weight[idx] = - math.log(1.0 / (ctr_avg) - 1.0)
                else:
                    weight[idx] = tool.next_init_weight()
        ctr = tool.sigmoid(value)
        return ctr