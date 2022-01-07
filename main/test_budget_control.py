#!/usr/bin/python
import sys
import random
import math
import argparse
import config
sys.path.append('../')
from base import tool
from base.tool import logger
from base import eval_tool
from bidding.bid_strategy import BidStrategy
from pacing.budget_pid_control import PID

def control_test(test_params, data_params):
    ecpcs = {}
    first_round = True
    sec_round = False
    cntr_size = int(len(data_params['yp']) / test_params['cntr_rounds'])
    total_cost = 0.0
    total_clks = 0
    total_wins = 0
    phi = 0
    logger.debug("round\tecpc\tphi\ttotal_click\tclick_ratio\twin_ratio\ttotal_cost\tecpc")
    pid = PID(test_params['ecpc'], first_round, sec_round, phi)
    for round in range(0, test_params['cntr_rounds']):
        pid.update(round, ecpcs)
        cost = 0
        clks = 0
        imp_index = ((round+1)*cntr_size)
        if round == test_params['cntr_rounds']- 1:
            imp_index = imp_index + (len(data_params['yp']) - cntr_size*test_params['cntr_rounds'])
        # phi bound
        phi = pid.check_bound()
        for i in range(round*cntr_size, imp_index):
            clk = data_params['y'][i]
            pctr = data_params['yp'][i]
            mp = data_params['mplist'][i]
            bid = max(test_params['minbid'], BidStrategy.bidding_lin(pctr, test_params['basectr'], test_params['basebid']) * (math.exp(phi)))
            if round == 0:
                bid = 1000.0
            if bid > mp:
                total_wins += 1
                clks += clk
                total_clks += clk
                cost += mp
                total_cost += mp
        ecpcs[round] = total_cost / (total_clks+1)
        click_ratio = total_clks * 1.0 / test_params['advs_test_clicks']
        win_ratio = total_wins * 1.0 / test_params['advs_test_bids']
        logger.debug("%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.1f" % (round, ecpcs[round], phi, total_clks, click_ratio, win_ratio, total_cost, test_params['ecpc']))
    report_params['overshoot'].append(eval_tool.cal_settling_time(ecpcs, test_params['ecpc'], test_params['settle_con'], test_params['cntr_rounds']))
    report_params['settling_time'].append(eval_tool.cal_settling_time(ecpcs, test_params['ecpc'], test_params['settle_con'], test_params['cntr_rounds']))
    result = eval_tool.cal_rise_time(ecpcs, test_params['ecpc'], test_params['rise_con'])
    report_params['rise_time'].append(result)
    report_params['rmse_ss'].append(eval_tool.cal_rmse_ss(ecpcs, test_params['ecpc'], test_params['settle_con'], test_params['cntr_rounds']))
    report_params['sd_ss'].append(eval_tool.cal_sd_ss(ecpcs, test_params['ecpc'], test_params['settle_con'], test_params['cntr_rounds']))
    return result

def readSample(sample_file, featWeight):
    labels = []
    mp_list = []
    pctr_list = []
    fi = open(sample_file, 'r')
    for line in fi:
        data = tool.ints(line.strip().replace(":1", "").split())
        clk = data[0]
        mp = data[1]
        fsid = 2 # feature start id
        feats = data[fsid:]
        pred = tool.estimate_ctr(featWeight, feats)
        labels.append(clk)
        pctr_list.append(pred)
        mp_list.append(mp)
    fi.close()
    return labels, mp_list, pctr_list

random.seed(10)

test_params = {
    'advs_test_bids' : 100000,
    'advs_test_clicks' : 65,
    'basebid' : 90,
    'basectr': 0.001,
    'minbid' : 5,
    'cntr_rounds' : 40,
    'para_gamma' : 60,
    'para_gammas' : range(60, 120, 5),
    'settle_con' : 0.1,
    'rise_con' : 0.9,
    'damping' : 0.25, 
    'para_p' : 0.0003,
    'para_i' : 0.000001,
    'para_d' : 0.0001,
    'div' : 1e-6,
    'para_ps' : range(0, 40, 5),
    'para_is' : range(0, 25, 5),
    'para_ds' : range(0, 25, 5),
    'min_phi' : -2,
    'max_phi' : 5,
    'ecpc' : 0
}

report_params = {
    'parameters' : [],
    'overshoot' : [],
    'settling_time' : [],
    'rise_time' : [],
    'rmse_ss' : [],
    'sd_ss' : []
}

def main():
    parser = argparse.ArgumentParser(description='please input cid(campaign id), budget')
    parser.add_argument('--cid', type=str, default="1458")
    parser.add_argument('--mode', type=str, default="batch or test")
    parser.add_argument('--ecpc', type=int, default=120000)
    args = parser.parse_args()
    advertiser = args.cid   
    test_params['ecpc'] = args.ecpc
    mode = args.mode

    data_params = {
        # parameter setting,   
        'train_path' : config.data_path+advertiser+"/train.yzx.txt",
        'test_path' : config.data_path+advertiser+"/test.yzx.txt",
        'model_path' : config.data_path+advertiser+"/train.yzx.txt.lr.weight",
        'model' : {},
        'y_train' : [],
        'mplist_train' : [],
        'yp_train' : [],
        'y' : [],
        'mplist' : [],
        'yp' : []
    }

    data_params['model'] = tool.load_lr_model(data_params['model_path'])
    data_params['y_train'], data_params['mplist_train'], data_params['yp_train'] = readSample(data_params['train_path'], data_params['model'])
    data_params['y'], data_params['mplist'], data_params['yp'] = readSample(data_params['test_path'], data_params['model'])
    test_params['basectr'] = sum(data_params['yp_train']) / float(len(data_params['yp_train']))
    parameter = ""+advertiser+"\t"+str(test_params['cntr_rounds'])+"\t"+str(test_params['basebid'])+"\t"+str(test_params['ecpc'])+"\t" + \
                str(test_params['para_p'])+"\t"+str(test_params['para_i'])+"\t"+str(test_params['para_d'])+"\t"+str(test_params['settle_con'])+"\t"+str(test_params['rise_con'])
    report_params['parameters'].append(parameter)
    if mode == "test": # test mode
        control_test(test_params, data_params)
        logger.debug("campaign\ttotal-rounds\tbase-bid\tecpc\tp\ti\td\tsettle-con\trise-con\trise-time\tsettling-time\tovershoot\trmse-ss\tsd-ss\n")
        for idx, val in enumerate(report_params['parameters']):
            logger.debug(val+"\t"+str(report_params['rise_time'][idx])+"\t"+str(report_params['settling_time'][idx])+"\t"+str(report_params['overshoot'][idx])+"\t" + \
                   str(report_params['rmse_ss'][idx]) + "\t" + str(report_params['sd_ss'][idx]))

if __name__ == '__main__':
    main()