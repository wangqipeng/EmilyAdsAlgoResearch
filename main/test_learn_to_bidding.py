#!/usr/bin/python
import os
import sys
import copy
import argparse
sys.path.append('../')
from base import tool
from base import eval_tool
from base.tool import logger
from configparser import ConfigParser
from base.dataset import Dataset
from market.bid_landscape import BidLandscape
from ctr.eu import EU
from ctr.rr import RR
from market.lin_market import LinMarket
from bidding.bidding_model import BiddingModel
from market.quad_market import QuadMarket

filename = os.path.basename(__file__).split('.')[0]
logger.add(filename+"_{time}.log")

def pre_train(train_data, test_data, ctr_model, mkt_model):
    ctr_model.train()
    ctr_model.test()
    logger.debug("Round 0"+ "\t" + str(tool.get_last_log(ctr_model.test_log)['performance']))

    mkt_model.train()
    train_anlp = eval_tool.calc_total_anlp(mkt_model, train_data)
    test_anlp = eval_tool.calc_total_anlp(mkt_model, test_data)
    logger.debug("Market Model pre-train ANLP train: %.3f, test: %.3f." % (train_anlp, test_anlp))

def main():
    parser = argparse.ArgumentParser(description='please input configre file')
    parser.add_argument('--config', type=str, default="config.ini")
    args = parser.parse_args()
    config_file = args.config
    cfg = ConfigParser()
    cfg.read(config_file)

    model_name = cfg.get('hyper_parameter', 'model_name')   
    mkt_model = cfg.get('hyper_parameter', 'mkt_model')
    data_folder = cfg.get('data', 'data_folder')
    campaign_id = cfg.getint('data', 'campaign_id')
    train_path = data_folder + str(campaign_id) + "/train.yzx.txt"
    test_path = data_folder + str(campaign_id) + "/test.yzx.txt"
    train_data = Dataset(train_path, campaign_id)
    train_data.shuffle() # make train data shuffled
    test_data = Dataset(test_path, campaign_id)
    if model_name == 'rr':
        ctr_model = RR(cfg, train_data, test_data)
    else:
        ctr_model = EU(cfg, train_data, test_data)
        
    if mkt_model == 'quad':
        mkt_model = QuadMarket(cfg, train_data, test_data)
    else:
        mkt_model = LinMarket(cfg, train_data, test_data)
    
    bid_model = BiddingModel(cfg, ctr_model, mkt_model, model_name)
    ctr_model.bid_strategy = bid_model.bid_strategy
    mkt_model.set_camp_v(ctr_model.camp_v) 
    mkt_model.set_ctr_model(ctr_model)
    mkt_model.set_bid_strategy(ctr_model.get_bid_strategy())
    logger.debug("campaign v = " + str(ctr_model.camp_v))
    
    bid_landscape = BidLandscape(train_data, train_data.get_camp_id())
    ctr_model.init_bid_landscape(bid_landscape)
  
    pre_train(train_data, test_data, ctr_model, mkt_model)

    logger.debug("Begin training ...")
    em_round = cfg.getint('hyper_parameter', 'em_round')
    eu_train_round = cfg.getint('hyper_parameter', 'eu_train_round')
    for i in range(0, em_round):#30
        recent_mkt_weight=[]
        recent_ctr_weight=[]
        for j in range(0, eu_train_round):#10
            #bid landscape model
            mkt_model.joint_train()
            test_anlp = eval_tool.calc_total_anlp(mkt_model, test_data)
            ctr_model.init_market_strategy(mkt_model)
            logger.debug("Tri Round " + str(i+1) + " Round " + str(j+1) + " test_anlp: " + str(test_anlp))
            #user reponse feedback
            ctr_model.train()
            ctr_model.test()
            logger.debug("Tri Round " + str(i+1) + " Round " + str(j+1) + "\t" + str(tool.get_last_log(ctr_model.test_log)['performance']))
            if j+1 > 3 :
                logger.debug("weight: ", len(recent_mkt_weight), len(recent_ctr_weight),len(mkt_model.weight), len(ctr_model.weight))
                del recent_mkt_weight[0]
                del recent_ctr_weight[0]
        recent_mkt_weight.append(copy.deepcopy(mkt_model.weight))
        recent_ctr_weight.append(copy.deepcopy(ctr_model.weight))
        mkt_model.weight = recent_mkt_weight[0]
        ctr_model.weight = recent_ctr_weight[0]
        #bid strategy model
        bid_model.train()
        logger.debug("Tri Round " + str(i+1) + "\t" + str(tool.get_last_log(bid_model.em_log)['performance']))
        if tool.judge_stop(bid_model.em_log):
            break
    logger.debug("Train done.")
    logger.debug("Best Result:")
    logger.debug(eval_tool.header)
    best_em_log = bid_model.get_best_log(bid_model.em_log)
    best_em_line = str(campaign_id) + "\t" + "em"+model_name + "\ttest\t" \
                    + tool.gen_performance_line(best_em_log) + "\t" \
                    + str(len(bid_model.em_log)) + "\t" + str(best_em_log['mu'])
    logger.debug(best_em_line)

if __name__ == '__main__':
    main()
