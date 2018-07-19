# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:50:56 2018

@author: Shinelon
"""
import numpy as np
import codecs
import os

class vote:
    def __init__(self):
        self.user = 0
        self.item = 0
        self.label = 0
        self.voteTime = 0

def fopen(path='hehe.txt', mode='r'):
    head = "C:/Users/Shinelon/Desktop/VBR2016"
    m = os.path.join(head, path)
    m = m.replace('\\','/')
    f = codecs.open(m, mode)
    return f
    
def inner(x, y):
    if not (len(x) == len(y)):
        print "inner error, size does not match"
    return np.dot(x, y)

def square(x):
    return x*x

def desquare(x):
    return 2*x

def stringTrim(s):
    return s.strip()

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


    
