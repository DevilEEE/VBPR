# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:14:30 2018

@author: Shinelon
"""
from VBR2016 import model
import numpy as np
import time
import copy

class BPRMF(model.model):
    def __init__(self, corp, K, lambd, biasReg):
        super(BPRMF, self).__init__(corp)
        self.K = K
        self.lambd = lambd
        self.biasReg = biasReg
        self.beta_item = []
        self.gamma_user = [[]]
        self.gamma_item = [[]]
        
    def init(self):
        self.NW = self.nItems + self.K*(self.nUsers + self.nItems)
        self.W = [0.0]*self.NW
        self.bestW = [0.0]*self.NW
        self.getParametersFromVectors(self.W, self.beta_item, self.gamma_user, self.gamma_item, 'INIT')
        return
    
    def cleanUp(self):
        self.getParametersFromVectors(self.W, self.beta_item, self.gamma_user, self.gamma_item, 'FREE')
        return
    
    def prediction(self, user, item):
        return self.beta_item[item] + np.dot(self.gamma_user[user], self.gamma_item[item])
    
    def getParametersFromVectors(self, g, beta_item, gamma_user, gamma_item, action='on'):
        if action == 'FREE':
            self.gamma_user = []
            self.gamma_item = []
            return
        if action == 'INIT':
            self.beta_item = g[:self.nItems]
            self.gamma_user = np.random.random((self.nUsers, self.K))
            self.gamma_item = np.random.random((self.nItems, self.K))
            return
        self.beta_item = g[:self.nItems]
        g = np.array(g[self.nItems:]).reshape(self.nUsers+self.nItems, self.K)
        self.gamma_user = g[:self.nUsers]
        self.gamma_item = g[self.nUsers:]
        return
    
    def sampleUser(self):
        while True:
            user_id = np.random.randint(0, self.nUsers-1)
            if len(self.pos_per_user[user_id]) == 0 or len(self.pos_per_user[user_id]) == self.nItems:
                continue
            return user_id
    
    def train(self, iterations, learn_rate):
        self.tostring1()
        bestValidAUC = -1
        best_iter = 0
        for Iter in range(iterations):
            clock_t = time.time()
            self.oneIteration(learn_rate)
            print "Iter: %d, took %f"%(Iter, time.time()-clock_t)
            if Iter % 5 == 0:
                self.AUC()
                print "[Valid AUC = %f], Test AUC = %f, Test Std = %f\n"%(self.AUC_val, self.AUC_test, self.std)
                if bestValidAUC < self.AUC_val:
                    bestValidAUC = self.AUC_val
                    best_iter = Iter
                    self.W = []
                    self.W.extend(self.beta_item)
                    self.W.extend(self.gamma_user.reshape(1,self.nUsers*self.K).tolist()[0]) 
                    self.W.extend(self.gamma_item.reshape(1,self.nItems*self.K).tolist()[0])
                    self.copyBestModel()
                elif self.AUC_val < bestValidAUC and Iter > best_iter + 50:
                    print "Overfitting!"
                    break
        #self.W = copy.deepcopy(self.bestW)
        self.getParametersFromVectors(self.bestW, self.beta_item, self.gamma_user, self.gamma_item, action='on')
        self.AUC()
        self.tostring2()
        return
    
    def oneIteration(self, learn_rate):
        print "oneIteration..."
        userMatrix = []
        for i in range(self.nUsers):
            userMatrix.append([])
        for u in range(self.nUsers):
            for w in self.pos_per_user[u]:
                userMatrix[u].append(w)
        for i in range(self.num_pos_events):
            if i%200 == 0:
                print i
            user_id = self.sampleUser()
            if len(userMatrix[user_id]) == 0:
                for w in self.pos_per_user[user_id]:
                    userMatrix[user_id].append(w)
            rand_num = np.random.randint(0, len(userMatrix[user_id]))
            pos_item_id = userMatrix[user_id][rand_num]
            userMatrix[user_id].remove(pos_item_id)
            while True:
                neg_item_id = np.random.randint(0, self.nItems-1)
                if not self.pos_per_user[user_id].has_key(neg_item_id):
                    break
            self.updateFactors(user_id, pos_item_id, neg_item_id, learn_rate)
        print "one iteration end!"
        return
    
    def updateFactors(self, user_id, pos_item_id, neg_item_id, learn_rate):
        #print "updateFactors..."
        x_uij = self.beta_item[pos_item_id] - self.beta_item[neg_item_id]
        x_uij += np.dot(self.gamma_user[user_id], self.gamma_item[pos_item_id]) - np.dot(self.gamma_user[user_id], self.gamma_item[neg_item_id])
        deri = 1.0/(1+np.exp(x_uij))
        self.beta_item[pos_item_id] += learn_rate * (deri - self.biasReg * self.beta_item[pos_item_id])
        self.beta_item[neg_item_id] += learn_rate * (-deri - self.biasReg * self.beta_item[neg_item_id])
        for f in range(self.K):
            w_uf = self.gamma_user[user_id][f]
            h_if = self.gamma_item[pos_item_id][f]
            h_jf = self.gamma_item[neg_item_id][f]
            self.gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - self.lambd * w_uf)
            self.gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - self.lambd * h_if)
            self.gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - self.lambd / 10.0 * h_jf)
        return
    
    def tostring1(self):
        print  "BPR-MF__K_%d_lambda_%.2f_biasReg_%.2f"%(self.K, self.lambd, self.biasReg)
        return
    
    def tostring2(self):
        print "<<< BPR-MF >>> Test AUC = %f, Test Std = %f\n"%(self.AUC_test, self.std)
        return
        