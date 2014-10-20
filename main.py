#!usr/bin/env python
#coding:utf-8

from plsa import CpLSA
from CFAlgorithm import CCFAlgorithm
import ReadData
import numpy as np

def getData(dataDict):
    usersSet = set()
    itemsSet = set()
    for user_id in dataDict:
        usersSet.add(user_id)
        for item_id, time in dataDict[user_id]:
            itemsSet.add(item_id)
    usersList = list(usersSet)
    itemsList = list(itemsSet)
    dataMatrix = np.zeros([usersList, itemsList])
    for user_id in dataDict:
        user_key = usersList.index(user_id)
        for item_id, time in dataDict[user_id]:
            item_key = itemsList.index(item_id)
            dataMatrix[user_key][item_key] = 1
    return usersList, itemsList, dataMatrix

def main():
    import ReadData
    filename = r'train_data.txt'
    print '.........读取数据...........'
    UsersItems, Items = ReadData.ReadData(filename)
    print '.........分割数据...........'
    train, test = ReadData.divideData(UsersItems)

    print '.............训练推荐............'
    flag = 0    #0: 表示训练测试， 1：表示生成结果
    near_num = 200; top_num = 1
    hiddenStates_num = 10; max_iter=30
    mPLSA = CpLSA()
    if flag==0:
        mPLSA.transformData(train)
        mPLSA.process(hiddenStates_num, max_iter)
        simUsers = mPLSA.calSimUsers(near_num)

    elif flag==1:
        mPLSA.transformData(UsersItems)
        mPLSA.process(hiddenStates_num, max_iter)

if __name__=='__main__':
    main()