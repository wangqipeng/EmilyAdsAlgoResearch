from ctr.lr import LR
from . opt_bid import OptBid
from market.lin_market import LinMarket
from ctr.eu import EU
from base.dataset import Dataset
import math
import random
import copy
import numpy as np
import base.eval_tool as eval_tool
import base.tool as tool
from base.tool import logger
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


class BiddingModel(LR):
    def __init__(self, config, ctr_model, mkt_model, model):
        LR.__init__(self, config, ctr_model.train_data, ctr_model.test_data)
        self.model = model
        self.em_log = []
        self.ctr_model=ctr_model
        self.mkt_model=mkt_model
        self.init_parameters()
        self.init_bid_strategy()

    def init_parameters(self):
        self.camp_v = self.train_data.get_statistics()['ecpc']
        self.budget = self.test_data.get_statistics()['cost_sum'] 
        self.mu = 0.0
        # budget is only used in test phase or M-step

    def init_bid_strategy(self):
        self.bid_strategy = OptBid(self.camp_v, self.mu)

    def train(self):
        self.m_step()

    def e_step(self):
        random.seed(10)
        train_data = self.train_data
        iter_id = train_data.init_index()
        while not train_data.reached_tail(iter_id):
            data = train_data.get_next_data(iter_id)
            y = data[0]
            feature = data[2:len(data)]
            ctr = self.ctr_model.estimate_ctr(self.ctr_model.weight, feature, train_flag = True)
            phi = 1.0 / (1.0 + self.mu)
            bp = self.bid_strategy.bid(ctr)
            pz = self.mkt_model.get_probability(bp, feature)
            scale_x = (phi * ctr - y) * phi * math.pow(self.camp_v, 2) * pz * config.em_scale
            if self.model == 'eu':
                scale_x = ctr * (1 - ctr) * scale_x

    def m_step(self):
        opt_mu = self.mu
        opt_revenue = -1E10
        opt_performance = {}
        test_data = self.test_data
        mu_range = np.arange(-0.99, 0.99, 0.01)
        for mu in mu_range:
            bid_sum = 0
            cost_sum = 0
            imp_sum = 0
            clk_sum = 0
            revenue_sum = 0
            labels = []
            p_labels = []
            self.bid_strategy.set_mu(mu)
            iter_id = test_data.init_index()
            while not test_data.reached_tail(iter_id):
                data = test_data.get_next_data(iter_id)
                bid_sum += 1
                y = data[0]
                mp = data[1]
                feature = data[2:len(data)]
                ctr = self.ctr_model.estimate_ctr(self.ctr_model.weight, feature, train_flag = True)
                labels.append(y)
                p_labels.append(ctr)
                bp = self.bid_strategy.bid(ctr)
                # bp = self.bid_strategy.bid(ctr)
                if bp > mp:
                    cost_sum += mp
                    imp_sum += 1
                    clk_sum += y
                    revenue_sum = int(revenue_sum - mp + y * self.camp_v * 1E3)
                if cost_sum >= self.budget:
                	break
            performance = eval_tool.get_performance(bid_sum, imp_sum, clk_sum, cost_sum, revenue_sum, labels, p_labels)
            #update bid strategy parameter
            if performance['revenue'] > opt_revenue:
                opt_revenue = performance['revenue']
                opt_performance = performance
                # slove the bidding function
                opt_mu = mu
        # reset the value of mu in both bidding function and model inner parameter
        self.bid_strategy.set_mu(opt_mu)
        self.mu = opt_mu
        log = self.make_log(self.ctr_model.weight, opt_performance)
        log['m'] = True
        self.test_log.append(log)
        self.em_log.append(log)

    def make_log(self, weight, performance):
        log = {}
        log['weight'] = copy.deepcopy(weight)
        log['performance'] = copy.deepcopy(performance)
        log['mu'] = self.mu
        return log

    def get_best_e_log(self, logs):
        best_log = {}
        if len(logs) == 0:
            logger.error("ERROR: no record in the log.")
        else:
            best_revenue = -1E10
            idx = len(logs)-1
            while idx>=0 and not 'm' in logs[idx]:
                log = logs[idx]
                revenue = log['performance']['revenue']
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_log = log
                idx -= 1
        return best_log
