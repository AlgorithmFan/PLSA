#!usr/bin/env python
#coding:utf-8

import codecs
import csv
def ReadData(filename):
    '''
    Read data from the train file.
    '''
    fp = codecs.open(filename, 'r', 'utf-8-sig')
    UsersItems = {}
    Items = {}
    for line in fp.readlines():
        temp = line.split()
        if len(temp)<5:
            continue
        user, item, rTime, title, content, nTime  = int(temp[0]), int(temp[1]), int(temp[2]), temp[3], temp[4:-1], temp[-1]
        #记录用户阅读过的News， UsersItems={u1:[(i1,t1), (i2, t2), ...], u2:[], u3:[], ....}
        if user not in UsersItems:
            UsersItems[user] = []
        UsersItems[user].append((item, rTime))
        #记录Items的Content，Items={i1：（），i2：（）， 。。。}
        if item not in Items:
            Items[item] = (title, content, nTime)
    fp.close()
    return UsersItems, Items

def divideData(data):
    '''
    将数据分割成为训练集和测试集
    '''
    train = {}
    test = {}
    for key in data:
        temp = data[key]
        temp.sort(key=lambda x:x[1])
        if key not in train:
            train[key] = temp[:-1]
            test[key] = temp[-2:]
    return train, test


def writeData(filename, recommendation):
    csvfile = file(filename, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['userid','newsid'])
    values = []
    for user in recommendation:
        for item in recommendation[user]:
            values.append((user, item))
    writer.writerows(values)
    csvfile.close()


if __name__=='__main__':
    filename = r'train_data.txt'
    UsersItems, Items = ReadData(filename)
    filename = 'see.txt'
    import time
    train, test = divideData(UsersItems)
    import codecs
    fp = codecs.open(filename, 'w','gbk')
    for user in UsersItems:
        if user==930:continue
        for item, r_time in UsersItems[user]:
            timeArray = time.localtime(r_time)

            fp.write('%s\t%s\t%s-%s-%s %s:%s:%s\t%s\t%s\n' %
                     (user, item, timeArray.tm_year, timeArray.tm_mon, timeArray.tm_mday, timeArray.tm_hour, timeArray.tm_min, timeArray.tm_sec, Items[item][0],
                     Items[item][2]))
    fp.close()