import math
import random
import copy
from model.model import Model

class MarketModel(Model):
	def __init__(self, config, train, test):
		Model.__init__(self, config, train, test)
		self.init_weight()

	def init_parameters(self):
		self.camp_v = self.train_data.get_statistics()['ecpc']
		self.test_log = []

	def init_weight(self):
		self.weight = {}
		self.best_weight = {}

	def set_camp_v(self, camp_v):
		self.camp_v = camp_v

	def set_ctr_model(self, ctr_model):
		self.ctr_model = ctr_model

	def set_bid_strategy(self, strategy):
		self.bid_strategy = strategy

	def get_probability(self, b, x):
		pass

	def get_win_probability(self, b, x):
		pass

	def calc_gradient_coeff(self, x, y, b):
		pass

	def update(self, x, y, b):
		pass

	def train(self):
		pass

	def joint_train(self):
		random.seed(10)
		train_data = self.train_data
		ctr_model = self.ctr_model
		iter_id = train_data.init_index()
		while not train_data.reached_tail(iter_id):
			data = train_data.get_next_data(iter_id)
			y = data[0]
			feature = data[2:len(data)]
			ctr = ctr_model.estimate_ctr(ctr_model.get_weight(), feature, train_flag=True, ctr_avg=ctr_model.train_data.get_statistics()['ctr'])
			bp = self.bid_strategy.bid(ctr)
			self.update(feature, y, bp)

	def test(self):
		anlp = self.calc_total_anlp(self.test_data)
		log = self.make_log(self.weight, anlp)
		self.test_log.append(log)
		return anlp

	def make_log(self, weight, anlp):
		log = {}
		log['weight'] = copy.deepcopy(weight)
		log['anlp'] = anlp
		return log