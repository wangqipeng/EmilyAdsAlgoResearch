#--- tool package ---#
import os
import random
import math
import time
from pathlib import Path
from loguru import logger
init_weight = 0.00011
# random.seed(10)

def next_init_weight():
	return (random.random() - 0.5) * init_weight

# convert string list to integer array [yzx]
def ints(data):
	int_array = []
	for d in data:
		int_array.append(int(d))
	return int_array

# convert to string list
def strings(data):
	str_array = []
	for d in data:
		str_array.append(str(d))
	return str_array

def to_str(bytes_or_str):
	if isinstance(bytes_or_str, bytes):
 		value = bytes_or_str.decode('utf-8')
	else:
 		value = bytes_or_str
	return value

def to_bytes(bytes_or_str):
	if isinstance(bytes_or_str, str):
 		value = bytes_or_str.encode('utf-8')
	else:
 		value = bytes_or_str
	return value # Instance of bytes

# sigmoid function
def sigmoid(z):
	value = 0.5
	try:
		value = 1.0 / (1.0 + math.exp(-z))
	except:
		value = 1E-9
	return value

def phi_t_x(phi, x, train_flag=False):
	result = 0.0
	for idx in x:
		if idx in phi:
			result += phi[idx]
		elif train_flag:
			phi[idx] = next_init_weight()
	return result

def estimate_ctr(weight, feature, train_flag = False):
	value = 0.0
	for idx in feature:
		if idx in weight:
			value += weight[idx]
		elif train_flag:
			weight[idx] = next_init_weight()
	ctr = sigmoid(value)
	return ctr

def calibrate_ctr(pctr, ds_ratio):
	cal_pctr = pctr / (pctr + (1 - pctr) / ds_ratio)
	return cal_pctr

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

def log_line(i, tri_log):
    return str(i+1) + "\t" + str(tri_log['performance']['revenue']) + "\t" \
                + str(tri_log['performance']['roi']) + "\t" \
                + str(tri_log['performance']['cpc']) + "\t" \
                + str(tri_log['performance']['ctr']) + "\t" \
                + str(tri_log['performance']['auc']) + "\t" \
                + str(tri_log['performance']['rmse']) + "\t" \
                + str(tri_log['performance']['imps']) + "\t" \
                + str(tri_log['weight'][0]) + "\t" \
		        + str(tri_log['mu'])

def judge_stop(logs):
	stop = False
	# step = int(1/config.train_progress_unit)
	step = 1
	curr_loop = len(logs) - 1 # the latest record id
	if curr_loop >= 2*step:
		current_r = logs[curr_loop]['performance']['revenue']
		last_r = logs[curr_loop - step]['performance']['revenue']
		last_2_r = logs[curr_loop - 2*step]['performance']['revenue']
		if current_r <= last_r and last_r <= last_2_r:
			stop = True
	return stop

def extend_judge_stop(logs):
	stop = False
	if len(logs) < 10:
		stop = False
	else:
		stop = judge_stop(logs)
	return stop

def get_last_log(logs):
	return logs[len(logs)-1]

def judge_file_exists(folder, file_name):
	if os.path.exists(file_name):
		return True
	else:
		return os.path.exists(folder + file_name)

#--- no use below ---#

# load data from file as [[yzx]]
def load_data(file_path):
	dataset = []
	if not os.path.isfile(file_path):
		logger.error("ERROR: file not exist. " + file_path)
	else:
		fi = open(file_path, 'r')
		for line in fi:
			li = ints(line.replace(':1','').split())
			dataset.append(li)
		fi.close()
	return dataset

def judge_market_stop(logs):
	stop = False
	step = 1
	curr_loop = len(logs) - 1 # the latest record id
	if curr_loop >= 2*step:
		current_r = logs[curr_loop]
		last_r = logs[curr_loop - step] 
		if last_r < 1E-15: 
			last_r = 1E-15
		last_2_r = logs[curr_loop - 2*step]
		dev = 1.0 * abs(current_r) - abs(last_r) / abs(last_r)
		if current_r > last_r and last_r > last_2_r or dev < 0.001:
			stop = True
	return stop

def output_weight(weight, path):
	if os.path.exists(path):
		logger.error("Warning: may override the existed file!")
	fo = open(path, 'w')
	for idx in weight:
		fo.write(str(idx) + '\t' + str(weight[idx]) + '\n')
	fo.close()

def load_lr_model(path):
	if not os.path.exists(path):
		logger.error("Error: The file does not exist!")
		exit(-1)
	weight = {}
	fi = open(path, 'r')
	for line in fi:
		str = line.split('\t')
		idx = int(str[0])
		value = float(str[1])
		weight[idx] = value
	fi.close()
	return weight

def output(arr, file_path): # array to file
	fo = open(file_path, 'a')
	for i in range(0,len(arr)):
		fo.write(str(arr[i]) + '\n')
	fo.close()
