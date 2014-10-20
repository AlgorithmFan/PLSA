#!usr/bin/env python
#coding:utf-8

import numpy as np

class CCFAlgorithm:
    def __init__(self, UserModels, ItemModels):
        self.UserModels = UserModels
        self.ItemModels = ItemModels


    def recommend(self, active_user, simUsers):
        '''Recommend items for this active_user.'''
        recommendation = {}
        for user_id in simUsers:
            temp_recommend = self.UserModels[active_user].calRecommendCF(self.UserModels[user_id])
            for item_id in temp_recommend:
                if item_id not in recommendation:
                    recommendation[item_id] = 0
                recommendation[item_id] += simUsers[user_id]
        return recommendation


    def trainProcess(self, simUsers, top_num, test=None):
        '''Train the data'''
        recommendation = {}
        for user_id in self.UserModels:
            temp_recommendation = self.recommend(user_id, simUsers[user_id])
            temp = sorted(temp_recommendation.iteritems(), key=lambda x:x[1], reverse=True)
            recommendation[user_id] = [news_id for news_id, sim in temp[:top_num]]
        return recommendation




