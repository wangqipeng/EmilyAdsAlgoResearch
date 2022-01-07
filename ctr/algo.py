import math
import random
from lifelines import KaplanMeierFitter

initWeight = 0.05
def nextInitWeight():
    return (random.random() - 0.5) * initWeight

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))

def ints(s):
    res = []
    for ss in s:
        res.append(int(ss))
    return res

def gdTrain(trainData, featWeight, lamb, eta):
    for data in trainData:
        clk = data[1]
        mp = data[2]
        fsid = 3 # feature start id
        # predict
        pred = 0.0
        for i in range(fsid, len(data)):
            feat = data[i]
            if feat not in featWeight:
                featWeight[feat] = nextInitWeight()
            pred += featWeight[feat]
        pred = sigmoid(pred)
        # start to update weight
        # w_i = w_i + learning_rate * [ (y - p) * x_i - lamb * w_i ] 
        for i in range(fsid, len(data)):
            feat = data[i]
            featWeight[feat] = featWeight[feat] * (1 - lamb) + eta * (clk - pred)
    return featWeight 

#T = [0, 3, 3, 2, 1, 2]  -  bid
#E = [1, 1, 0, 0, 1, 1]  -  win or not
#kmf.fit(T, event_observed=E)
def survival_analysis(bids, results):
    kmf = KaplanMeierFitter()
    kmf.fit(bids, results)
    return kmf

def win_prob(bid, kmf):
    return 1 - kmf.survival_function_['KM_estimate'][bid]

def getBids(data):
    pass

def getBidResults(data):
    pass

def gdTrainUnbias(trainData, featWeight, lamb, eta):
    kmf = survival_analysis(getBids(trainData), getBidResults(trainData))
    for data in trainData:
        clk = data[1]
        mp = data[2]
        w = win_prob(mp, kmf)
        fsid = 3 # feature start id
        # predict
        pred = 0.0
        for i in range(fsid, len(data)):
            feat = data[i]
            if feat not in featWeight:
                featWeight[feat] = nextInitWeight()
            pred += featWeight[feat]
        pred = sigmoid(pred)
        # start to update weight
        # w_i = w_i + learning_rate * [ (y - p) * x_i - lamb * w_i ]
        ws = 1
        importance_pow = 1
        importance = math.pow(ws / w, importance_pow)
        #print str(importance)
        for i in range(fsid, len(data)):
            feat = data[i]
            featWeight[feat] = featWeight[feat] * (1 - (eta * importance) * lamb) + (eta * importance) * (clk - pred)
    return featWeight

