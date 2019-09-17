# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:14:22 2019

@author: Anmin Zou
"""

import pandas as pd
import numpy as np


"""
读取已经处理过的的数据
"""

path = "D:\\Work at STU\\Data\\adult.csv"
data = pd.read_csv(path)

D = data.isnull().sum(axis=1)
E = data.isnull().sum(axis=0)
print('Name of features containing missing values:',
      E[E > 0].index) #存在缺失值的特征

print('Total number of missing values:',
      sum(E)) #缺失值的总数

num = len(D[D > 0])
num1 = len(D[D == 1])
num2 = len(D[D == 2])
num3 = len(D[D == 3])

print('Total number of samples containing missing values:',
      num) #存在缺失值的样本数目
print('Total number of samples containing one missing value:', 
      num1) #只有一个特征存在缺失值的样本数目

print('Total number of samples containing two missing values simultaneously:', 
      num2) #两个特征同时存在缺失值的样本数目

print('Total number of samples containing three missing values simultaneously:',
      num3) #三个特征同时存在缺失值的样本数目

m, n = data.shape
print('Percentage of samples containing missing values (%):', 100*np.round(num/m, 4))
print('Percentage of high income samples in Adult Data Set (%):', 
      100*np.round(len(data.y[data.y == 1])/m, 4))

index = D[D > 0].index
y_ms = data.y[index] 
"""
y_ms: Response variables of samples with missing values
"""
print('Percentage of high income samples in the sample set with missing values (%):', 
      100*np.round(len(y_ms[y_ms == 1])/len(y_ms), 4))

y_nms = data.y[D[D == 0].index]
"""
y_nms: Response variables of samples without missing values
"""
print('Percentage of high income samples in the sample set without missing values (%):', 
      100*np.round(len(y_nms[y_nms == 1])/len(y_nms), 4))

"""
1. 整个数据集的样本数足够大，m = 32561；
2. 包含缺失值的样本的比例是：2399/32561 = 7.37%；
3. 整个数据集中正负样本的比例是：24.08%:75.92%； 
   去掉含缺失值样本的数据集中的正负样本比例是：24.89%:75.11%. 
   （包含缺失值的样本删除掉并没有明显影响正负样本的比例）
"""

data1 = data.dropna(axis = 0) 
"""
axis = 0: 删除带有缺失值的行
axis = 1: 删除带有缺失值的列
"""
m1, n1 = data1.shape
print('m1 =', m1)
print('n1 =', n1)

# 将数据写入文本格式
path = "D:\\Work at STU\\Data\\adult1.csv"
data1.to_csv(path, na_rep = 'NA', index = False, header = True)