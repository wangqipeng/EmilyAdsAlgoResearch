# -*- coding: utf-8 -*-
import sys
import argparse
import config
from pandas import read_csv
import base.tool as tool
from pmdarima.arima import auto_arima
sys.path.append('../')
from base import tool
from bidding.bid_strategy import BidStrategy
from pacing.budget_pacing import BudgetPacing

DATA_FOLDER="/Users/ad/Doc/Project/make-ipinyou-data/"
TRAIN_PREFIX="flow_model_train_"
TEST_PREFIX="flow_model_test_"
POSTFIX=".csv"

params = {
    "seasonal": True,
    "start_p": 0,
    "start_q": 0,
    "max_p": 5,
    "max_q": 5,
    "m": 7,
}

def read_tyz(sample_file, featWeight):
    labels = []
    mp_list = []
    pctr_list = []
    ts_list = []
    fi = open(sample_file, 'r')
    for line in fi:
        data = tool.ints(line.strip().replace(":1", "").split())
        ts = data[0]
        clk = data[1]
        mp = data[2]
        fsid = 2 # feature start id
        feats = data[fsid:]
        pred = tool.estimate_ctr(featWeight, feats)
        ts_list.append(ts)
        labels.append(clk)
        pctr_list.append(pred)
        mp_list.append(mp)
    fi.close()
    return ts_list, labels, mp_list, pctr_list

def try_bidding(basebid, pctr, mp, basectr, budget, total_cost, slot_cost):
    minbid = 5
    pBid = BidStrategy.bidding_lin(pctr, basectr, basebid)
    bid = max(minbid, pBid)
    if bid > mp:
        total_cost += mp
        slot_cost += mp
        budget -= mp
    return total_cost, slot_cost, budget

#对于冷启动场景，预算平滑可先验的在各个slot预分配预算。一个slot结束以后，视消耗情况对余下slot的预算做再分配
#为了避免每个slot的预算，很快花完，应该设定一个pacing rate 
#即 1.根据历史数据预估下一个slot的预算
def main():
    parser = argparse.ArgumentParser(description='please input campaign id')
    parser.add_argument('--cid', type=str, default="campaign id")
    parser.add_argument('--budget', type=int, default="20000")
    args = parser.parse_args()
    cid = args.cid
    budget = args.budget
    #initialize the lr
    featWeight = {}
    model_path = DATA_FOLDER+cid+"/train.yzx.txt.lr.weight"
    featWeight = tool.load_lr_model(model_path)
    train_file = DATA_FOLDER+cid+"/train_pacing.txt"
    test_file = DATA_FOLDER+cid+"/test_pacing.txt"
    _, _, _, yp_train = read_tyz(train_file, featWeight)
    ts_list, y, mplist, yp = read_tyz(test_file, featWeight)
    basectr = sum(yp_train) / float(len(yp_train))
    basebid = (sum(mplist) / float(len(mplist)))
	#flow sample format:
	#format: timestap | PV
    #00|4193
    #15|2376
    #30|1146
    #45|1298
    train = read_csv(DATA_FOLDER+cid+"/"+TRAIN_PREFIX+cid+POSTFIX, sep="|",header=0, parse_dates=True, index_col=0, squeeze=True)
    test = read_csv(DATA_FOLDER+cid+"/"+TEST_PREFIX+cid+POSTFIX, sep="|",header=0, parse_dates=True, index_col=0, squeeze=True)
    test = test.values.tolist()
    pacing = BudgetPacing(train, params, budget, ts_list[0])
    pacing.fit()
    #96 15 minutes
    cnt = 0
    total_cost = 0
    slot_cost = 0
    slot_budget = pacing.predict_budget(test, cnt)
    print('{:>12}  {:>12}  {:>12}  {:>12} {:>12}'.format('total_cost', 'budget_remaining', 'slot_cost', 'slot_budget', 'ptr'))
    for i, ts in enumerate(ts_list):
        is_need = pacing.is_need_update(ts, slot_cost)
        #is_need, current = is_need_update(ts, current)
        if is_need:
            cnt+=1
            if cnt == 96:#one day:24*4(00,15,30,45)
                break
            print(f'{total_cost:>12}  {budget:>12}  {slot_cost:>12}  {slot_budget[0]:>12} {pacing.get_ptr() :>12}')
            slot_budget = pacing.update(budget, slot_cost, slot_budget[0], test, cnt)
            slot_cost = 0
        if  pacing.is_try_bidding():
            total_cost, slot_cost, budget = try_bidding(basebid, yp[i], mplist[i], basectr, budget, total_cost, slot_cost)
    
if __name__ == '__main__':
    main()