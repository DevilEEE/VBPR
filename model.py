# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:26:20 2018

@author: Shinelon
"""
from VBR2016 import common, corpus
import numpy as np
import copy

class model(object):
    def __init__(self, corp):
        self.corp = corp
        self.nUsers = self.corp.nUsers
        self.nItems = self.corp.nItems
        self.nVotes = self.corp.nVotes
        self.test_per_user = [(-1, -1)]*self.nUsers
        self.val_per_user = [(-1, -1)]*self.nUsers
        self.pos_per_user = []
        self.pos_per_item = []
        for i in range(self.nUsers):
            self.pos_per_user.append({})
        for i in range(self.nItems):
            self.pos_per_item.append({})
        for i in range(self.nVotes):
            user = self.corp.V[i].user
            item = self.corp.V[i].item
            voteTime = self.corp.V[i].voteTime
            if self.test_per_user[user][0] == -1:
                self.test_per_user[user] = (item, voteTime)
            elif self.val_per_user[user][0] == -1:
                self.val_per_user[user] = (item, voteTime)
            else:
                self.pos_per_user[user][item] = voteTime
                self.pos_per_item[item][user] = voteTime
        
        self.num_pos_events = 0
        for i in range(self.nUsers):
            self.num_pos_events += len(self.pos_per_user[i])
        #模型参数
        #self.NW = 0
        #self.W = []
        #self.bestW = []
        
        self.itemPrice = {}
        self.itemBrand = {}
        
    def AUC(self):
        AUC_u_val = [0]*self.nUsers
        AUC_u_test = [0]*self.nUsers
        for u in range(self.nUsers):
            item_test = self.test_per_user[u][0]
            item_val = self.val_per_user[u][0]
            x_u_test = self.Aprediction(u, item_test)
            x_u_val = self.Aprediction(u, item_val)
            count_test = 0
            count_val = 0
            maxnum = 0
            for j in range(self.nItems):
                if (not self.pos_per_user[u].has_key(j)) or item_test == j or item_val == j:
                    continue
                maxnum += 1
                x_uj = self.Aprediction(u, j)
                if x_u_test > x_uj:
                    count_test += 1
                if x_u_val > x_uj:
                    count_val += 1
            try:
                AUC_u_val[u] = 1.0*count_val/maxnum
                AUC_u_test[u] = 1.0*count_test/maxnum
            except:
                print count_val, count_test, maxnum, x_u_test, x_u_val
        self.AUC_val = sum(AUC_u_val)/self.nUsers
        self.AUC_test = sum(AUC_u_test)/self.nUsers
        self.std = np.std(AUC_u_test)
        return 
    
    def AUC_codeItem(self, AUC_test, std, num_user):
        return
    
    def copyBestModel(self):
        self.bestW = copy.deepcopy(self.W)
        return
    
    def saveModel(self, path):
        f = common.fopen(path, 'w')
        self.stringBestW = [str(w)+' ' for w in self.bestW]
        f.writelines(self.stringBestW)
        f.close()
        return
    
    def loadModel(self, path):
        f = common.fopen(path, 'r')
        self.stringBestW = ''
        for line in f.readlines():
            self.stringBestW += line
        self.bestW = [int(w.strip()) for w in self.stringBestW.split()]
        f.close()
        return
    
    def toString(self):
        return "Empty Model!"
    
    def Aprediction(self, user, item):
        self.childPrediction = getattr(self, 'prediction')
        return self.childPrediction(user, item)
        