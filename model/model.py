#!/usr/bin/python
from base.tool import logger
from base.dataset import Dataset

class Model:
    def __init__(self, config, train, test):
        self.set_config(config)
        self.train_data = train
        self.test_data = test

    def set_config(self, cfg):
        self.data_folder = cfg.get('data', 'data_folder')
        self.campaign_id = cfg.getint('data', 'campaign_id')
        self.model_name = cfg.get('hyper_parameter', 'model_name')   
        self.mkt_model = cfg.get('hyper_parameter', 'mkt_model')
        self.laplace = cfg.getint('hyper_parameter', 'laplace')
        self.eu_scale = cfg.getint('hyper_parameter', 'eu_scale')
        self.ds_ratio = cfg.getfloat('hyper_parameter', 'ds_ratio') if float(cfg.get('hyper_parameter', 'ds_ratio'))>0 else 0
        self.budget_prop = cfg.getfloat('hyper_parameter', 'budget_prop')
        self.lr_alpha = cfg.getfloat('hyper_parameter', 'lr_alpha')
        self.use_market = cfg.getboolean('market_parameter', 'use_market')
        self.market_lambda = cfg.getfloat('market_parameter', 'market_lambda')
        self.market_alpha = cfg.getfloat('market_parameter', 'market_alpha')
        logger.debug("cam_id\tmodel\tlaplace\tscale\tds_ratio\tbudget_prop\tmkt_alpha\tmkt_lambda")
        logger.debug(str(self.campaign_id) + "\t" + str(self.model_name) \
					+ "\t" + str(self.laplace) + "\t" + str(self.eu_scale) \
					+ "\t" + str(self.ds_ratio) + "\t" + str(self.budget_prop) \
					+ "\t" + str(self.market_alpha) + "\t" + str(self.market_lambda))

    def set_train_data(self, train_data):
        self.train_data = train_data

    def set_test_data(self, test_data):
        self.test_data = test_data

    def output_weight(self, weight, path):
        fo = open(path, 'w')
        for idx in weight:
            fo.write(str(idx) + '\t' + str(weight[idx]) + '\n')
        fo.close()

    def fit(self):
        pass

    def converged(self):
        pass

    def test(self):
        pass

    def calc_performance(self, dataset):
        pass