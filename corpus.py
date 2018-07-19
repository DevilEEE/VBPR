# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from VBR2016 import common
import random
import struct

class corpus:
    def __init__(self):
        self.V = []
        self.nUsers = 0
        self.nItems = 0
        self.nVotes = 0
        self.userIds = {}
        self.itemIds = {}
        self.ruserIds = {}
        self.ritemIds = {}
        self.imgAsins = {}
        self.uCounts = {}
        self.bCounts = {}
        self.imageFeatures = {}
        self.imageFeatureDim = 4096
        
    def loadData(self, voteFile, imgFeatPath, userMin, itemMin):
        self.nUsers = 0
        self.nItems = 0
        self.nVotes = 0
        self.imageFeatureDim = 4096
        
        self.loadVotes(imgFeatPath, voteFile, userMin, itemMin)
        self.loadImageFeatures(imgFeatPath)
        print "\n  \"nUsers\": %d, \"nItems\": %d, \"nVotes\": %d\n"%(self.nUsers, self.nItems, self.nVotes)
        return 
    
    def cleanUp(self):
        self.V = []
        return 
    
    def loadVotes(self, imgFeatPath, voteFile, userMin, itemMin):
        f = common.fopen(imgFeatPath, 'rb')
        print "pre-loading image asins from %s"%(imgFeatPath)
        #feat = [0.0]*self.imageFeatureDim
        while True:
            asin = f.read(10)
            #print asin
            if asin == '':
                break
            asin == asin.strip()
            self.imgAsins[asin] = 1
            feature = []
            for i in range(4096):
                feature.append(struct.unpack('f', f.read(4)))
        f.close()
        
        print "Loading votes from %s, userMin = %d, itemMin = %d  "%(voteFile, userMin, itemMin)
        self.voteMap = {}
        f1 = common.fopen(voteFile)
        count = 0
        for l in f1:
            l = l.strip()
            l = l.split(',')
            uName = l[0]
            bName = l[1]
            value = l[2]
            count += 1
            if count % 10000 == 0:
                print count
            if not self.imgAsins.has_key(bName):
                continue
            if not self.uCounts.has_key(uName):
                self.uCounts[uName] = 0   
            if not self.bCounts.has_key(bName):
                self.bCounts[bName] = 0
            self.uCounts[uName] += 1
            self.bCounts[bName] += 1
        f1.close()
        self.nUsers = 0
        self.nItems = 0
        f2 = common.fopen(voteFile)
        count = 0
        for l in f2:
            l = l.strip()
            l = l.split(',')
            count += 1
            if count % 10000 == 0:
                print count
            uName, bName, value, voteTime = l[0], l[1], l[2], l[3]
            if not self.imgAsins.has_key(bName):
                continue
            if self.uCounts[uName] < userMin or self.bCounts[bName] < itemMin:
                continue
            if not self.itemIds.has_key(bName):
                self.ritemIds[self.nItems] = bName
                self.itemIds[bName] = self.nItems
                self.nItems += 1
            if not self.userIds.has_key(uName):
                self.ruserIds[self.nUsers] = uName
                self.userIds[uName] = self.nUsers
                self.nUsers += 1
            self.voteMap[(self.userIds[uName], self.itemIds[bName])] = voteTime
        f2.close()
        self.generateVotes()
        return
    
    def loadImageFeatures(self, imgFeatPath):
        f = common.fopen(imgFeatPath, 'rb')
        print "\nLoading imgFeatures from %s"%imgFeatPath
        ma = 58.388599
        while True:
            asin = f.read(10)
            if asin == '':
                break
            asin = asin.strip()
            feature = []
            for i in range(4096):
                feature.append(struct.unpack('f', f.read(4)))
            if not self.itemIds.has_key(asin):
                continue
            for i in range(4096):
                feature[i] = feature[i] + (feature[i][0]/ma,)
            self.imageFeatures[self.itemIds[asin]] = feature
        f.close()
        return
    
    def generateVotes(self):
        print "\n Generating votes data: "
        for key in self.voteMap:
            v = common.vote()
            v.user = key[0]
            v.item = key[1]
            v.voteTime = self.voteMap[key]
            self.V.append(v)
        self.nVotes = len(self.V)
        random.shuffle(self.V)
        return