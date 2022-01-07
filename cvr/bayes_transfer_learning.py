import math
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

class BayesTransferLearning(object):
    def __init__(self, train_data, test_data, src_model, penalty = "L2", gamma = 0, fit_intercept = True):
        self.loadLRModel(src_model)
        self.init(train_data, test_data)
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.e = 1e-6
        self.alpha = 0.001
        self.tao_init = 100
        self.tao = self.tao_init
        self.N = 0 
        self.M = 0
        self.beta  = np.random.rand(self.train_X.shape[1])#csr_matrix((1, self.length), dtype=np.float64) #
        self.g_i   = csr_matrix((1, self.length), dtype=np.float64)
        self.v_i   = csr_matrix((1, self.length), dtype=np.float64)
        self.h_i   = csr_matrix((1, self.length), dtype=np.float64)
        self.tao_i = csr_matrix(np.ones((1, self.length)))
        self.eta   = csr_matrix((1, self.length), dtype=np.float64)
    
    def loadLRModel(self, lr_model):
        with open(lr_model, 'r') as m:
            lines = m.readlines()
            feat_num, _ = lines[-1].strip().split('\t')
            self.length = int(feat_num)+1
            self.u = [0]*(self.length)
            for line in lines:
                fi, w = line.strip().split('\t')
                self.u[int(fi)] = float(w)

    def batchBayesTransLearning(self, X, y, u, max_iter=1000, lr=0.01, tol=1e-7):
        if self.fit_intercept :
            X = np.c_[np.ones(X.shape[0]), X]
        
        pre_loss = np.inf
        self.beta = np.rand(X.shape[1])
        for _ in max_iter:
            y_pred = math.sigmoid(np.dot(X, self.beta))
            loss = self.batchLoss(X, y, y_pred)
            if loss - pre_loss < tol:
                return
            pre_loss = loss
            self.beta -= lr * batchGradient(X, y, y_pred, u)

    def batchLoss(self, X, y, y_pred, u):
        N, M = X.shape
        order = 2 if self.penalty == "L2" else 1
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        penalty = 0.5 * self.gamma * np.linalg.norm(self.beta - u, ord=order) ** 2
        return (penalty + nll) / N 

    def calcPenalty(self, X, u):
        N, M = X.shape
        p = self.penalty
        g = self.gamma
        b = self.beta
        l1norm = lambda x: np.linalg.norm(x, 1)
        delta = np.array(b) - np.array(u)
        return delta * g if p == "L2" else g * l1norm(delta) * np.sign(delta)

    def gradientWithPrior(self, X, y, y_pred, u):
        penalty = self.calcPenalty(X, u)
        return -(np.dot((y - y_pred), X) + penalty)/self.N

    def gradient(self, X, y, y_pred):
        """Gradient of the penalized negative log likelihood wrt beta"""
        N, M = X.shape
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)
        return -(np.dot(y - y_pred, X) + d_penalty) / N 

    def newton(self, X, y_pred, u):
        penalty = self.calcPenalty(X, u)      
        return (np.dot(np.dot(X, X.T), np.dot(y_pred, (1-y_pred))) + penalty)/self.N

    def hessianWithPrior(self, X, y_pred, u):
        penalty = self.calcPenalty(X, u)     
        return (np.dot(X, X.T) * np.diag(y_pred) * np.diag(1 - y_pred)+penalty)/self.N  

    def hessian(self, X, y_pred):
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)    
        return (np.dot(X, X.T) * np.diag(y_pred) * np.diag(1 - y_pred)+d_penalty)/self.N  

    def drawTrainSample(self):
        i = random.randint(0, self.train_X.shape[0])
        return self.train_X.getrow(i), self.train_y[i]

    def drawTestSample(self):
        i = random.randint(0, self.test_X.shape[0])
        return self.test_X.getrow(i), self.test_y[i]
    
    def featFreqStat(self, indices):
        featDict = {}
        for idx in indices:
            if idx in featDict:
                featDict[idx]+=1
            else:
                featDict[idx] = 1
        return featDict

    def init(self, train, test):
        self.train_X = train[0]
        self.train_y = train[1] 
        self.test_X = test[0]
        self.test_y = test[1]
        self.Fn = self.featFreqStat(self.train_X.indices)
    
    def predict_prob(self, X):
        z = X.dot(self.beta.transpose())
        y_pred = expit(z.astype(dtype=np.float64))
        return y_pred

    #Hyper-parameter free learning
    def fit(self):
        for k in range(30):
            X, y = self.drawTrainSample()
            self.N, self.M = X.shape
            z = X.dot(self.beta.transpose())
            y_pred = expit(z.astype(dtype=np.float64))
            g = self.gradient(X, y, y_pred)
            h = self.hessian(X, y_pred)
            for i in range(self.M):#for each dimension
                if X[0,i] == 0:
                    continue
                self.g_i[0,i] = self.g_i[0,i]*(1 - 1/self.tao_i[0,i]) + g[i][0,i]*(1/self.tao_i[0,i])
                self.v_i[0,i] = self.v_i[0,i]*(1 - 1/self.tao_i[0,i]) + g[i][0,i]*g[i][0,i]*(1/self.tao_i[0,i])
                self.h_i[0,i] = self.h_i[0,i]*(1 - 1/self.tao_i[0,i]) + h[0][i]*(1/self.tao_i[0,i])
                reg = np.dot(np.dot(self.gamma, (self.N/self.Fn[i])), (self.beta[i] - self.u[i]))
                if self.tao_i[0,i] < self.tao_init:
                    self.beta[i] -= self.e *(g[i][0,i] + reg)
                    self.tao_i[0,i] += 1 
                else:
                    ratio = ((self.g_i[0,i] + reg)*(self.g_i[0,i] + reg))/(self.v_i[0,i] + 2*self.g_i[0,i]*reg + reg*reg)  
                    self.eta[i] = ratio / (self.h_i[0,i] + self.gamma*(self.N/self.Fn[i]))
                    self.beta[i] -= self.eta[i] * (g[i][0,i] + reg)
                    self.tao_i[0,i] += (1 - ratio)*self.tao_i[0,i] + 1
            val_X, val_y = self.drawTestSample()
            val_z = val_X.dot(self.beta.transpose())
            val_y_pred = expit(val_z.astype(dtype=np.float64))
            val_g = self.gradient(val_X, val_y, val_y_pred)
            product = val_g.dot(self.beta)
            sum_g = product.sum()
            self.gamma += self.alpha * self.gamma * sum_g   
            
                  