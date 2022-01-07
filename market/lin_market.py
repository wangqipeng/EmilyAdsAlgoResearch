import base.tool as tool
import numpy as np
from model.model import Model
from base.tool import logger
from . market_model import MarketModel


class LinMarket(MarketModel):
	def __init__(self, config, train_data, test_data):
		Model.__init__(self, config, train_data, test_data)
		self.init_parameters()
		self.init_weight()
		self.market_alpha = self.market_alpha
		self.market_lambda = self.market_lambda
		self.reg_update_param = 1 - self.market_alpha * self.market_lambda
		logger.debug("Linear Market Model camp_v: " + str(self.camp_v))
		logger.debug("reg_update_param: " + str(self.reg_update_param))

	def get_probability(self, z, x):
		result = 1E-10
		exp = tool.phi_t_x(self.weight, x)
		ma = np.exp(exp) 
		if z > ma:
			result = 1E-10
		else:
			result = 1.0 / ma
		return result

	def get_win_probability(self, b, x):
		result = 1E-50
		if b == 0:
			result = 1E-50
		else:
			exp = tool.phi_t_x(self.weight, x)
			ma = np.exp(exp) 
			if b > ma:
				result = 1.0
			else:
				result = 1.0 * b / ma
		return result

	def calc_gradient_coeff(self, x, y, b, train_flag=True):
		exp = tool.phi_t_x(self.weight, x, train_flag)
		exp_result = np.exp(exp) 
		coeff = (b * b / 2 - self.camp_v * y * b) * (-1.0) / exp_result
		return coeff

	def update(self, x, y, b):
		gradient_coeff = self.calc_gradient_coeff(x, y, b, train_flag=True)
		for idx in x:
			if idx not in self.weight:
				self.weight[idx] = tool.next_init_weight()
			else:
				self.weight[idx] = self.weight[idx] * self.reg_update_param - self.market_alpha * gradient_coeff # * 1 of weight[idx]

	# Note: This is the self-train function, with ground truth market price z.
	def train(self):
		idx = self.train_data.init_index()
		while not self.train_data.reached_tail(idx):
			data = self.train_data.get_next_data(idx)
			y = data[0]
			z = data[1]
			x = data[2:len(data)]
			self.update(x, y, z)

	def load_weight(self, path):
		self.weight = tool.load_lr_model(path)
		if self.weight == None:
			logger.error("Error: load weight error!")
			exit(-1)
		else:
			return True