# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:06:21 2018

@author: Shinelon
"""
from VBR2016 import BPRMF
import numpy as np

class MMMF(BPRMF.BPRMF):
    def __init__(self, corp, K, lambd, biasReg):
        super(MMMF, self).__init__(corp, K, lambd, biasReg)
        
    def updataFactors(self, user_id, pos_item_id, neg_item_id, learn_rate):
        x_uij = self.beta_item[pos_item_id] - self.beta_item[neg_item_id]
        x_uij += np.dot(self.gamma_user[user_id], self.gamma_item[pos_item_id]) - np.dot(self.gamma_user[user_id], self.gamma_item[neg_item_id])
        deri = 1.0/(1+np.exp(x_uij))
        if x_uij < 0:
            deri = 1
        else:
            deri = 0
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
        print  "MMMF__K_%d_lambda_%.2f_biasReg_%.2f"%(self.K, self.lambd, self.biasReg)
        return
    
    def tostring2(self):
        print "<<< MMMF >>> Test AUC = %f, Test Std = %f\n"%(self.AUC_test, self.std)
        return
