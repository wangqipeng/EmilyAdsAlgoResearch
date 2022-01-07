from math import *
import os
from copy import deepcopy
import numpy as np

UPPER = 301
# s is a list(or dict of lists) of market price: market_price - num
# normal version
# s: feature_payprice - num
def calProbDistribution_n(s,minPrice,maxPrice,info):
    laplace = info.laplace
    q = [0.0]*UPPER
    count = 0
    if isinstance(s,dict):
        for k in s.keys():#each feature val
            for i in range(0,len(s[k])):#
                q[i] += s[k][i]
                count += s[k][i]
    elif isinstance(s,list):
        for i in range(0,len(s)):
            q[i] += s[i]#
            count += s[i] #total
    #probability of the distribution
    for i in range(0,len(q)):
        q[i] = (q[i]+laplace)/(count+len(q)*laplace)        #laplace smooth
    return q

def lowBound(a,b):
    return a/(a+b) - 1.65*np.sqrt((a*b)/( (a+b)**2*(a+b+1)))

# uniform the probability in each bucket (to smooth the survival probability)(this function is not used now)
def changeBucketUniform(x, step):
    if step==1:
        return deepcopy(x)
    bucket_num = int(ceil(float(len(x))/step))
    y = [0.]*len(x)
    for i in range(0, bucket_num):
        bsum = 0.
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            bsum += x[j]
        for j in range(i*step,(i+1)*step):
            if j>=len(x):
                break
            y[j] = bsum/step
    return y
    
#_P:feature_price - num
def KLDivergence(_P, _Q,step = STEP):
    if len(_P)!=len(_Q):
        print('kld:',len(_P),len(_Q))
        return 0
    P = changeBucketUniform(_P,step)
    Q = changeBucketUniform(_Q,step)
    KLD = 0.0
    for i in range(0,len(P)):
        if Q[i] == 0:
            continue
        KLD += P[i]*log(P[i]/Q[i])

    return KLD

def pearsonr(x, y):
    if len(x)!=len(y):
        print('len(x) != len(y)')
        return
    meanx = float(sum(x))/len(x)
    meany = float(sum(y))/len(y)
    sx = 0.
    sy = 0.
    for i in range(0,len(x)):
        sx += (x[i]-meanx)**2
        sy += (y[i]-meany)**2
    sx = sx**0.5
    sy = sy**0.5

    r = 0.
    for i in range(0,len(x)):
        r += (x[i]-meanx)*(y[i]-meany)
    r /= sx*sy

    return r

# cross entropy = avg( y[i] log yp[i] + (1 - y[i]) log(1 - yp[i]) )
# entropy = - (avg(y) log avg(y) + (1 - avg(y)) log(1 - avg(y)))
# rig = entropy + cross_entropy
def get_relative_information_gain(y, yp):
    for i in range(len(yp)):
        if yp[i] < 1E-8:
            yp[i] = 1E-8
        elif yp[i] > 1 - 1E-8:
            yp[i] = 1 - 1E-8

    ce = get_cross_entropy(y, yp)
    p_avg = np.average(y)
    h = - (p_avg * np.log2(p_avg) + (1 - p_avg) * np.log2(1 - p_avg))
    ig = ce + h
    rig = ig / h
    return rig

def get_cross_entropy(y, yp):
    y = np.array(y)
    yp = np.array(yp)
    ce = 0.
    for i in range(len(yp)):
        ce += - y[i] * np.log2(yp[i]) - (1- y[i]) * np.log2(1 - yp[i])
    ce = ce / len(yp)
    return ce