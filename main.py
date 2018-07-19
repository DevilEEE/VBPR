# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 13:05:41 2018

@author: Shinelon
"""
from VBR2016 import BPRMF, corpus, MMMF, VBPR
corp = corpus.corpus()
corp.loadData('ratings_Video_Games.csv','image_features_Video_Games.b',15,15)
def goMMMF(corp, K, lambd, learn_rate, iterations, biasReg):
    x = MMMF.MMMF(corp, K, lambd, biasReg)
    x.init()
    x.train(iterations, learn_rate)
    x.saveModel('hehe.txt')
    return

def goBPRMF(corp, K, lambd, learn_rate, iterations, biasReg):
    x = BPRMF.BPRMF(corp, K, lambd, biasReg)
    x.init()
    x.train(iterations, learn_rate)
    x.saveModel('hehe.txt')
    return

def goVBPR(corp, K, K2, lambd, lambd2, biasReg, iterations, learn_rate):
    x = VBPR.VBPR(corp, K, K2, lambd, lambd2, biasReg)
    x.init()
    x.train(iterations, learn_rate)
    x.saveModel('hehe.txt')
    return

goBPRMF(corp, 20, 10, 0.01, 20, 0.01)
goVBPR(corp, 20, 20, 10, 10, 0.01, 5, 0.01)
