# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:42:39 2018

@author: Shinelon
"""
from VBR2016 import BPRMF
import numpy as np
import random
import time

class VBPR(BPRMF.BPRMF):
    def __init__(self, corp, K, K2, lambd, lambd2, biasReg):
        super(VBPR, self).__init__(corp, K, lambd, biasReg)
        self.K2 = K2
        self.lambd2 = lambd2
    
    def init(self):
        self.NW = self.K * self.nUsers + (self.K + 1) * self.nItems + self.K2 * self.nUsers + self.K2 * self.corp.imageFeatureDim + self.corp.imageFeatureDim
        self.W = np.zeros((1, self.NW))
        self.bestW = np.zeros((1, self.NW))
        for i in range(self.nItems, self.NW-self.corp.imageFeatureDim):
            self.W[0][i] = random.random()
        self.getParametersFromVector(self.W, 'INIT')
        return
    
    def cleanUp(self):
        self.getParametersFromVector(self.W, 'FREE')
        self.W = []
        self.bestW = []
        return
    
    def getParametersFromVector(self, g, action):
        if action == 'FREE':
            self.gamma_user = []
            self.gamma_item = []
            self.theta_user = []
            self.U = []
            return
        if action == 'INIT':
            self.beta_item = np.zeros((1, self.nItems)).tolist()[0]
            self.gamma_user = np.ones((self.nUsers, self.K)).tolist()
            self.gamma_item = np.ones((self.nItems, self.K)).tolist()
            self.theta_user = np.ones((self.nUsers, self.K2)).tolist()
            self.U = np.ones((self.K2, self.corp.imageFeatureDim)).tolist()
            self.beta_cnn = np.zeros((1, self.corp.imageFeatureDim)).tolist()[0]
            self.theta_item = np.zeros((self.nItems, self.K2)).tolist()
            self.beta_item_visual = np.zeros((1, self.nItems)).tolist()[0]
        return
    
    def getVisualFactors(self):
        self.theta_item = np.zeros((self.nItems, self.K2)).tolist()
        self.beta_item_visual = np.zeros((1, self.nItems)).tolist()[0]
        for x in range(self.nItems):
            feat = self.corp.imageFeatures[x]
            for k in range(self.K2):
                for i in range(len(feat)):
                    self.theta_item[x][k] += self.U[k][feat[i][0]]*feat[i][1]
            for i in range(len(feat)):
                self.beta_item_visual[x] += self.beta_cnn[feat[i][0]]*feat[i][1]
        return
    
    def prediction(self, user, item):
        return self.beta_item[item] + np.dot(self.gamma_user[user], self.gamma_item[item]) + np.dot(self.theta_item[item], self.theta_user[user]) + self.beta_item_visual[item]
    
    def train(self, iterations, learn_rate):
        self.tostring1()
        bestValidAUC = -1
        best_iter = 0
        for Iter in range(iterations):
            clock_t = time.time()
            self.oneIteration(learn_rate)
            print "Iter: %d, took %f"%(Iter, time.time()-clock_t)
            if Iter % 1 == 0:
                self.getVisualFactors()
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
                elif self.AUC_val < bestValidAUC and Iter > best_iter + 20:
                    print "Overfitting!"
                    break
        #self.W = copy.deepcopy(self.bestW)
        self.getParametersFromVectors(self.bestW, action='on')
        self.AUC()
        self.tostring2()
        return
    
    def updateFactors(self, user_id, pos_item_id, neg_item_id, learn_rate):
        #print "updataFactors..."
        diff = []
        feat_i = self.corp.imageFeatures[pos_item_id]
        feat_j = self.corp.imageFeatures[neg_item_id]
        p_i = 0
        p_j = 0
        while p_i < len(feat_i) and p_j < len(feat_j):
            ind_i = int(feat_i[p_i][0])
            ind_j = int(feat_j[p_j][0])
            if ind_i < ind_j:
                diff.append((ind_i, feat_i[p_i][1]))
                p_i += 1
            elif ind_i > ind_j:
                diff.append((ind_j, -feat_j[p_j][1]))
                p_j += 1
            else:
                diff.append((ind_i, feat_i[p_i][1]-feat_j[p_j][1]))
                p_i += 1
                p_j += 1
        while p_i < len(feat_i):
            diff.append((int(feat_i[p_i][0]), feat_i[p_i][1]))
            p_i += 1
        while p_j < len(feat_j):
            diff.append((int(feat_j[p_j][0]), -feat_j[p_j][1]))
            p_j += 1
        for r in range(self.K2):
            self.theta_item[0][r] = 0
            for ind in range(len(diff)):
                c = diff[ind][0]
                self.theta_item[0][r] += self.U[r][c]*diff[ind][1]
        visual_bias = 0
        for ind in range(len(diff)):
            c = diff[ind][0]
            visual_bias += self.beta_cnn[c]*diff[ind][1]
        x_uij = self.beta_item[pos_item_id] - self.beta_item[neg_item_id]
        x_uij += np.dot(self.gamma_user[user_id], self.gamma_item[pos_item_id]) - np.dot(self.gamma_user[user_id], self.gamma_item[neg_item_id])
        x_uij += np.dot(self.theta_user[user_id], self.theta_item[0])
        x_uij += visual_bias
        deri = 1./(1+np.exp(x_uij))
        self.beta_item[pos_item_id] += learn_rate * (deri - self.biasReg * self.beta_item[pos_item_id])
        self.beta_item[neg_item_id] += learn_rate * (-deri - self.biasReg * self.beta_item[neg_item_id])
        for f in range(self.K):
            w_uf = self.gamma_user[user_id][f]
            h_if = self.gamma_item[pos_item_id][f]
            h_jf = self.gamma_item[neg_item_id][f]
            self.gamma_user[user_id][f]     += learn_rate * ( deri * (h_if - h_jf) - self.lambd * w_uf)
            self.gamma_item[pos_item_id][f] += learn_rate * ( deri * w_uf - self.lambd * h_if)
            self.gamma_item[neg_item_id][f] += learn_rate * (-deri * w_uf - self.lambd / 10.0 * h_jf)
        for f in range(self.K2):
            for ind in range(len(diff)):
                c = diff[ind][0]
                self.U[f][c] += learn_rate * (deri * self.theta_user[user_id][f] * diff[ind][1] - self.lambd2 * self.U[f][c])
            self.theta_user[user_id][f] += learn_rate * (deri * self.theta_item[0][f] - self.lambd * self.theta_user[user_id][f])
        for ind in range(len(diff)):
            c = diff[ind][0]
            self.beta_cnn[c] += learn_rate * (deri * diff[ind][1] - self.lambd2 * self.beta_cnn[c])
        #print "updataFactors..."
        return
    
    def tostring1(self):
        print  "VBPRF__K_%d_lambda_%.2f_biasReg_%.2f"%(self.K, self.lambd, self.biasReg)
        return
    
    def tostring2(self):
        print "<<< VBPRF >>> Test AUC = %f, Test Std = %f\n"%(self.AUC_test, self.std)
        return
    
    
