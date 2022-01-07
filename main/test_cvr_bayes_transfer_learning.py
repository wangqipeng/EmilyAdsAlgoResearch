#!/usr/bin/python
import os
import sys
import copy
import argparse
sys.path.append('../')
from base import eval_tool
from base.tool import logger
from configparser import ConfigParser
from base.dataset import Dataset
from sklearn.datasets import load_svmlight_file
from cvr.bayes_transfer_learning import BayesTransferLearning

def main():
    parser = argparse.ArgumentParser(description='please input cvr configre file')
    parser.add_argument('--config', type=str, default="cvr_config.ini")
    args = parser.parse_args()
    config_file = args.config
    cfg = ConfigParser()
    cfg.read(config_file)
    data_folder = cfg.get('data', 'data_folder')
    campaign_id = cfg.getint('data', 'campaign_id')
    train_path = data_folder + str(campaign_id) + "/train_libsvm.txt"
    valid_path = data_folder + str(campaign_id) + "/valid_libsvm.txt"
    test_path  = data_folder + str(campaign_id) + "/test_libsvm.txt"
    src_model_path = data_folder + str(campaign_id) + "/train.yzx.txt.lr.weight"
    sp_train_data = load_svmlight_file(train_path, n_features = 560870)
    sp_valid_data = load_svmlight_file(valid_path, n_features = 560870)
    sp_test_data = load_svmlight_file(test_path, n_features = 560870)
    cvr_model = BayesTransferLearning(sp_train_data, sp_valid_data, src_model_path)
    cvr_model.fit()
    preds = cvr_model.predict_prob(sp_test_data[0])
    print("Test AUC: ", eval_tool.roc_auc_score(sp_test_data[1], preds))
    

if __name__ == '__main__':
    main()