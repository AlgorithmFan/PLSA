'''
Author: Zhang Haidong
E-mail:haidong_zhang@163.com
A pLSA for collaborative filtering was proposed by Hofmann. This program is mainly according to his paper.
'''

#!usr/bin/env python
#coding:utf-8
import numpy as np
import cPickle
from scipy.sparse import lil_matrix
from normalize import normalizeVec, normalizeMatrix
class CpLSA:
    def __init__(self):
        self.dataMatrix = None
        self.hiddenStates_num = 0
        self.user_num, self.item_num = 0, 0
        self.usersList, self.itemsList = [], []

    def transformData(self, dataDict):
        usersSet = set()
        itemsSet = set()
        #Get the users set and items set.
        for user_id in dataDict:
            usersSet.add(user_id)
            for item_id, time in dataDict[user_id]:
                itemsSet.add(item_id)

        #
        self.usersList = list(usersSet)
        self.itemsList = list(itemsSet)
        self.user_num = len(self.usersList)
        self.item_num = len(self.itemsList)
        #Data Matrix
        self.dataMatrix = np.zeros([self.user_num, self.item_num], dtype=np.int)
        for user_id in dataDict:
            user_key = self.usersList.index(user_id)
            for item_id, time in dataDict[user_id]:
                item_key = self.itemsList.index(item_id)
                self.dataMatrix[user_key][item_key] = 1

    def calSimUsers(self, near_num):
        '''Calculate the similar users.'''
        simUsers = {}
        for user_key in range(self.user_num):
            active_user = self.usersList[user_key]
            temp_simUsers = {}
            for another_user_key in range(self.user_num):
                another_user = self.usersList[another_user_key]
                if active_user == another_user:
                    continue
                temp_simUsers[another_user] = self.calSimilarity(self.user_hidden_prob[user_key], self.user_hidden_prob[another_user_key])
            temp_simUsers = sorted(temp_simUsers.iteritems(), key=lambda x:x[1], reverse=True)
            simUsers[active_user] = {user:sim for user, sim in temp_simUsers[:near_num]}
        return simUsers

    def calSimilarity(self, userA, userB):
        '''Calculate the similarity between users'''
        userADet = sum(userA)
        userBDet = sum(userB)
        return userA * userB/(userADet*userBDet)


    def calSimItems(self):
        '''Calculate the similar items.'''


    def initParameters(self, hiddenStates_num):
        #Initialize the parameters of pLSA
        #self.data = data    #
        self.hiddenStates_num = hiddenStates_num
#        self.user_num, self.item_num = np.shape(self.dataMatrix)
        self.user_hidden_prob = np.random.random(size=(self.user_num, self.hiddenStates_num)) #P(z|u)
        normalizeMatrix(self.user_hidden_prob)
        self.hidden_item_prob = np.random.random(size=(self.hiddenStates_num, self.item_num)) #P(y|z)
        normalizeMatrix(self.hidden_item_prob)
        self.hidden_prob = np.zeros([self.user_num, self.item_num, self.hiddenStates_num], dtype=np.float) #Q(z,u,y)

    def process(self, hiddenStates_num, max_iter):
        '''
        Train the data
        '''
        self.initParameters(hiddenStates_num)
        for iteration in range(max_iter):
            print 'Iteration %s:' % str(iteration+1)
            print '\tE Step:'
            for user_key in range(self.user_num):
                for item_key in range(self.item_num):
                    prob = self.user_hidden_prob[user_key, :] * self.hidden_item_prob[:, item_key]
                    normalizeVec(prob)
                    self.hidden_prob[user_key][item_key] = prob
            print '\tM Step:'
            #Update p(z|u)
            for user_key in range(self.user_num):
                for hidden_key in range(self.hiddenStates_num):
                    sum = 0.0
                    for item_key in range(self.item_num):
                        sum += self.dataMatrix[user_key][item_key]*self.hidden_prob[user_key][item_key][hidden_key]
                    self.user_hidden_prob[user_key][hidden_key] = sum
                normalizeVec(self.user_hidden_prob[user_key])

            #Update p(y|z)
            for hidden_key in range(self.hiddenStates_num):
                for item_key in range(self.item_num):
                    sum = 0.0
                    for user_key in range(self.user_num):
                        sum += self.dataMatrix[user_key][item_key]*self.hidden_prob[user_key][item_key][hidden_key]
                    self.hidden_item_prob[hidden_key][item_key] = sum
                normalizeVec(self.hidden_item_prob[hidden_key])
        fp = open('user_hidden_prob.txt')
        cPickle.dump(self.user_hidden_prob, fp)
        fp.close()