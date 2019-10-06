# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:47:14 2019

@author: lenovo
"""

import numpy as np
import pandas as pd

#knn算法
def knn(train_set, train_label, test_set, k):
    ##利用欧式距离计算最近邻
    #生成ndarray类型的数组，用于存放训练样本和测试样本之间的欧式距离
    distance_arr = np.empty(len(train_set))  
    #遍历训练集中的每一个数据，计算训练集中每个数据与测试数据的欧氏距离
    for i in range(len(train_set)):
        #计算训练数据与测试数据距离的平方和
        dis = (train_set[i][0] - test_set[0]) ** 2 + (train_set[i][1] - test_set[1]) ** 2
        #将距离的平方和开根号
        dis = dis ** 0.5
        #添加到欧式距离数组中
        distance_arr[i] = dis

    #数组值从小到大的索引值排序，获得排序后原始数据下角标
    index = distance_arr.argsort()  
    #获得距离最小的前k个近邻的下角标
    min_index = index[:k]  
    
    ##计算前k个近邻中每个类别的个数
    #生成字典类型，用于存放类别出现的次数
    count_label = {}  
    #遍历前k个近邻的类别
    for i in min_index:
        label = train_label[i]
        #若数组中存在该类别，则类别数加1
        if label in count_label:
            count_label[label] = count_label[label] + 1
        #否则将类别添加到字典中，初始值为1
        else:
            count_label[label] = 1
        
    #使用sorted函数对labelCount按照value值降序排序
    count_sorted=sorted(count_label.items(), key=lambda d:d[1], reverse=True)  

    #返回标签出现最多的那个为预测类别
    return count_sorted[0][0]  

#根据训练集对测试集的类别进行预测
def predict(train_set, train_label, test_set, k):
    label = []
    for i in range(len(test_set)):
        test_label = knn(train_set, train_label, test_set[i], k)
        label.append(test_label)
    return label

#读取训练集和测试集数据
train = np.loadtxt('HTRU_2_train.csv', delimiter=",")
train_set = train[:, :-1]
train_label = train[:, -1]
test_set = np.loadtxt('HTRU_2_test.csv', delimiter=",")

#设定k的值
k = 15

#进行类别预测
test_predict = predict(train_set, train_label, test_set, k)
#print(test_predict)

#保存成csv格式
test_predict = pd.DataFrame(test_predict)
test_predict.index += 1
test_predict.to_csv('predict2.1.csv')


#正确率 0.85714


