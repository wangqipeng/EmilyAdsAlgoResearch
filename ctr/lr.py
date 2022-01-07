#!/usr/bin/python
import sys
sys.path.append('../')
from base.tool import logger
import copy
import random
from model.model import Model
from bidding.opt_bid import OptBid
import base.tool as tool
import base.eval_tool as eval_tool
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

class LR(Model):
	def __init__(self, config, train, test):
		Model.__init__(self, config, train, test)
		self.init_parameters()
		self.init_weight()
		self.init_bid_strategy()
		self.lr_alpha = 5E-3 
		self.lr_lambda = 1E-4
		self.reg_update_param = 1 - self.lr_alpha * self.lr_lambda
		self.train_log = []
		self.test_log = []

	def init_weight(self):
		self.weight = {}
		self.best_weight = {}

	def init_bid_strategy(self):
		self.bid_strategy = OptBid(self.camp_v)

	def init_parameters(self):
		self.camp_v = self.train_data.get_statistics()['ecpc']
		self.mu = 0.0
		self.budget = self.test_data.get_statistics()['cost_sum'] 

	def get_weight(self):
		if self.weight == None:
			logger.error("ERROR: Please init the CTR model weight!")
		return self.weight

	def get_bid_strategy(self):
		if self.bid_strategy == None:
			logger.error("ERROR: Please init bid strategy first.")
		return self.bid_strategy

	def train(self): # train with one traversal of the full train_data
		random.seed(10)
		train_data = self.train_data
# 		print "Train data \t" + `train_data` + "\tsize \t" + `train_data.get_size()`
		iter_id = train_data.init_index()
		while not train_data.reached_tail(iter_id):
			data = train_data.get_next_data(iter_id)
			y = data[0]
			feature = data[2:len(data)]
			ctr = tool.estimate_ctr(self.weight, feature, train_flag=True)
			for idx in feature: # update
				self.weight[idx] = self.weight[idx] * self.reg_update_param - self.lr_alpha * (ctr - y)

	def test(self):
		parameters = {'weight':self.weight}
		performance = self.calc_performance(self.test_data, parameters)
		# record performance
		log = self.make_log(self.weight, performance)
		self.test_log.append(log)

	def make_log(self, weight, performance):
		log = {}
		log['weight'] = copy.deepcopy(weight)
		log['performance'] = copy.deepcopy(performance)
		log['mu'] = self.mu
		return log

	def calc_performance(self, dataset, parameters): # calculate the performance w.r.t. the given dataset and parameters
		weight = parameters['weight']
		# budget = parameters['budget']
		bid_sum = 0
		cost_sum = 0
		imp_sum = 0
		clk_sum = 0
		revenue_sum = 0
		labels = []
		p_labels = []
		iter_id = dataset.init_index()
		while not dataset.reached_tail(iter_id): #TODO no budget set
			bid_sum += 1
			data = dataset.get_next_data(iter_id)
			y = data[0]
			market_price = data[1]
			feature = data[2:len(data)]
			ctr = tool.estimate_ctr(weight, feature, train_flag=False)
			labels.append(y)
			p_labels.append(ctr)
			bid_price = self.bid_strategy.bid(ctr)
			if bid_price > market_price:
				cost_sum += market_price
				imp_sum += 1
				clk_sum += y
				revenue_sum = int(revenue_sum - market_price + y * self.camp_v * 1E3)
			if cost_sum >= self.budget:
				break
		return eval_tool.get_performance(bid_sum, imp_sum, clk_sum, cost_sum, revenue_sum, labels, p_labels)

	def get_best_log(self, logs):
		best_log = {}
		if len(logs) == 0:
			logger.error(("ERROR: no record in the log."))
			exit()
		else:
			best_revenue = -1E30
			for log in logs:
				revenue = log['performance']['revenue']
				if revenue > best_revenue:
					best_revenue = revenue
					best_log = log
		return best_log

	def lin_bid(self, weight):
		params = range(30, 100, 5) + range(100, 400, 10) + range(400, 800, 50)
		base_ctr = self.train_data.get_statistics()['ctr']
		dataset = self.test_data
		opt_param = 3000
		opt_revenue = -1E10
		for param in params:
			bid_sum = 0
			cost_sum = 0
			imp_sum = 0
			clk_sum = 0
			revenue_sum = 0
			labels = []
			p_labels = []
			iter_id = dataset.init_index()
			while not dataset.reached_tail(iter_id): #TODO no budget set
				bid_sum += 1
				data = dataset.get_next_data(iter_id)
				y = data[0]
				market_price = data[1]
				feature = data[2:len(data)]
				ctr = tool.estimate_ctr(weight, feature, train_flag=False)
				labels.append(y)
				p_labels.append(ctr)
				bid_price = int(param * ctr / base_ctr)
				if bid_price > market_price:
					cost_sum += market_price
					imp_sum += 1
					clk_sum += y
					revenue_sum = int(revenue_sum - market_price + y * self.camp_v * 1E3)
				if cost_sum >= self.budget:
					break
			performance = eval_tool.get_performance(bid_sum, imp_sum, clk_sum, cost_sum, revenue_sum, labels, p_labels)
			if performance['revenue'] > opt_revenue:
				opt_revenue = performance['revenue']
				opt_param = param
		self.opt_param = opt_param
		return opt_param

	def replay(self, weight, test_data, budget_prop):
		budget = int(1.0 * test_data.get_statistics()['cost_sum'] / budget_prop)
		base_ctr = self.train_data.get_statistics()['ctr']
		label = []
		p_labels = []
		bid_sum = 0
		cost_sum = 0
		imp_sum = 0
		clk_sum = 0
		revenue_sum = 0
		labels = []
		p_labels = []
		iter_id = test_data.init_index()
		while not test_data.reached_tail(iter_id):
			data = test_data.get_next_data(iter_id)
			bid_sum += 1
			y = data[0]
			mp = data[1]
			feature = data[2:len(data)]
			ctr = tool.estimate_ctr(weight, feature, train_flag=False)
			labels.append(y)
			p_labels.append(ctr)
			bp = int(self.opt_param * ctr / base_ctr)
			# bp = self.bid_strategy.bid(ctr)
			if bp > mp:
				cost_sum += mp
				imp_sum += 1
				clk_sum += y
				revenue_sum = int(revenue_sum - mp + y * self.camp_v * 1E3)
			if cost_sum >= budget:
				break
		return eval_tool.get_performance(bid_sum, imp_sum, clk_sum, cost_sum, revenue_sum, labels, p_labels)
