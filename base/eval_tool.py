import parser
import math
import numpy as np
from math import log2
from math import log
from base.tool import logger
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
#from sklearn.utils import check_arrays
header = "camp_id\tmodel\tdataset\trevenue\troi\tctr\tcpc\tauc\trmse\tcpm\tbids\timps\tclks\tlaplace\tinterval\tscale\tds_ratio\tbudget_prop\tem_round\tmu"


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    error = np.sum(np.abs(y_true - y_pred))
    total = np.sum(y_pred)*len(y_true)
    return error/total

def get_performance(bid_sum, imp_sum, clk_sum, cost_sum, revenue_sum, labels, p_labels):
	cpc = 0.0 if clk_sum == 0 else 1.0 * cost_sum / clk_sum * 1E-3
	cpm = 0.0 if imp_sum == 0 else 1.0 * cost_sum / imp_sum
	ctr = 0.0 if imp_sum == 0 else 1.0 * clk_sum / imp_sum
	roi = 0.0 if cost_sum == 0 else 1.0 * revenue_sum / cost_sum
	auc = roc_auc_score(labels, p_labels)
	rmse = math.sqrt(mean_squared_error(labels, p_labels))
	performance = {'bids':bid_sum, 'cpc':cpc, 'cpm':cpm, 
					'auc': auc, 'rmse': rmse,
					'ctr': ctr, 'revenue':revenue_sum, 
					'imps':imp_sum, 'clks':clk_sum,
					'roi': roi}
	return performance

def gen_performance_line(log):
	performance = log['performance']
	line = str(performance['revenue']) + "\t" \
			+ str(performance['roi']) + "\t" \
			+ str(performance['ctr']) + "\t" \
			+ str(performance['cpc']) + "\t" \
			+ str(performance['auc']) + "\t" \
			+ str(performance['rmse']) + "\t" \
			+ str(performance['cpm']) + "\t" \
			+ str(performance['bids']) + "\t" \
			+ str(performance['imps']) + "\t" \
 			+ str(performance['clks'])
	return line

def getANLP(q,n,minPrice,maxPrice):
    anlp = 0.0
    N = 0
    if isinstance(q,dict):
        for k in n.keys():
            if not q.has_key(k):
                continue
            for i in range(0,len(n[k])):
                if i > len(q[k])-1:   # test price doesn't appear in train price
                    break    # remain to be modify
                anlp += -log(q[k][i])*n[k][i] # the sum of log probability for all sample data Pnl
				#Pij means the probability of training sample in the ith leaf node given price j
				#Nij is the number of test sample in the ith leaf node given price j
                N += n[k][i]
    if isinstance(q,list):
        for i in range(0,len(n)):
            if i > len(q)-1:
                break
            anlp += -log(q[i])*n[i] # the sum of log probability for all sample data Pnl
            N += n[i]
    anlp = anlp/N
    return anlp,N

def calc_nlp(mkt, x, z):
	prob = mkt.get_probability(z,x)
	if prob <= 0:
		return "ERROR_INVALID"
	nlp = -1.0 * math.log(prob)
	return nlp

def calc_total_anlp(mkt, dataset):
	idx = dataset.init_index()
	anlp = 0.0
	counter = 0
	while not dataset.reached_tail(idx):
		counter += 1
		data = dataset.get_next_data(idx)
		z = data[1]
		x = data[2:len(data)]
		res = calc_nlp(mkt, x, z)
		if res == "ERROR_INVALID":
			continue
		anlp += res
		anlp /= counter
	return anlp

# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

def get_best_kpi():
    pass

# calculate percentage overshoot
def cal_overshoot(winrs, ref):
    if winrs[0] > ref:
        min = winrs[0];
        for key, value in winrs.tems():
            if value <= min:
                min = value
        if min < ref:
            return (ref - min) * 100.0 / ref
        else:
            return 0.0
    elif winrs[0] < ref:
        max = winrs[0]
        for key, value in winrs.items():
            if value >= max:
                max = value
        if max > ref:
            return (max - ref) * 100.0 / ref
        else:
            return 0.0
    else:
        max = 0
        for key, value in winrs.items():
            if abs(value - ref) >= max:
                max = value
        return (max - ref) * 100.0 / ref

def cal_settling_time(winrs, ref, settle_con, cntr_rounds):
    settled = False
    settling_time = 0
    for key, value in winrs.items():
        error = ref - value
        if abs(error) / ref <= settle_con and settled == False:
            settled = True
            settling_time = key
        elif abs(error) / ref > settle_con:
            settled = False
            settling_time = cntr_rounds
    return settling_time

# # calculate steady-state error
def cal_rmse_ss(winrs, ref, settle_con, cntr_rounds):
    settling_time = cal_settling_time(winrs, ref, settle_con, cntr_rounds)
    rmse = 0.0
    if settling_time >= cntr_rounds:
        settling_time = cntr_rounds - 1
    for round in range(settling_time, cntr_rounds):
        rmse += (winrs[round] - ref) * (winrs[round] - ref)
    rmse /= (cntr_rounds - settling_time)
    rmse = math.sqrt(rmse) / ref
    return rmse

# # calculate steady-state standard deviation
def cal_sd_ss(winrs, ref, settle_con, cntr_rounds):
    settling_time = cal_settling_time(winrs, ref, settle_con, cntr_rounds)
    if settling_time >= cntr_rounds:
        settling_time = cntr_rounds - 1
    sum2 = 0.0
    sum = 0.0
    for round in range(settling_time, cntr_rounds):
        sum2 += winrs[round] * winrs[round]
        sum += winrs[round]
    n = cntr_rounds - settling_time
    mean = sum / n
    sd = math.sqrt(sum2 / n - mean * mean) / mean # weinan: relative sd
    return sd

# calculate rise time
def cal_rise_time(winrs, ref, rise_con):
    rise_time = 0
    for key, value in winrs.items():
        error = ref - value
        if abs(error) / ref <= (1 - rise_con):
            rise_time = key
            break
    return rise_time