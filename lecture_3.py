# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:49:11 2019

@author: Anmin Zou
"""

import pandas as pd
import numpy as np
from numpy import nan as NA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
读取已经处理过的的数据
"""

path = "D:\\Work at STU\\Data\\adult1.csv"
data = pd.read_csv(path)

Cnames = data.columns

m, n = data.shape

"""
制作提供信息的可视化（有时称为绘图）是数据分析中的最重要任务之一。
可视化是探索性过程的一部分，例如可以帮助识别异常值或所需的数据转换，或者为建模提供一些想法。
matplotlib（www.matplotlib.org/）用于制图及其它二位数据可视化的Python库。
"""

"""
不同收入水平的受教育年限箱形图。
"""
s1 = data["edu-num"][data.y == 0]
Edu_num0 = data["edu-num"][data.y == 0] 
Edu_num1 = data["edu-num"][data.y == 1]

plt.figure(1)
plt.boxplot(Edu_num0)
plt.xticks([1],  ["0"])
plt.ylabel("Education Number")
plt.xlabel("Low Income")
plt.grid(axis = "y", ls = ":", lw = 1, color = "gray", alpha = 0.4)
plt.show()

plt.figure(2)
plt.boxplot(Edu_num1)
plt.xticks([1], ["1"])
plt.ylabel("Education Number")
plt.xlabel("High Income")
plt.grid(axis = "y", ls = ":", lw = 1, color = "gray", alpha = 0.4)
plt.show()

"""
交叉报表是一种常用的数据分析方法。它可以描述两个变量之间的关系。
"""
from statsmodels.graphics.mosaicplot import mosaic

"""
受教育程度和收入水平的交叉报表。
"""
cross1 = pd.crosstab(pd.qcut(data["edu-num"], [0, .25, .5, .75, 1]), 
                     data.y)
"""
pd.qcut(): 基于分位数(Quantile)的离散化函数。
pd.crosstab(): 计算两个（或多个）因素的简单交叉表。
"""
print(cross1)
mosaic(cross1.stack(), gap = 0.02) 
"""
mosaic(): 可视化多变量类别数据。
"""

"""
每周工作时间和收入水平的交叉报表。
"""
cross2 = pd.crosstab(pd.cut(data["hpw"], 5), data.y)
print(cross2)
#将交叉报表归一化，利于分析数据
cross2_norm = cross2.div(cross2.sum(1).astype(float), axis = 0)
cross2_norm.plot(kind = "bar")
plt.show()

"""
年龄和收入水平的交叉报表。
"""
cross3 = pd.crosstab(pd.cut(data["age"], [17, 25, 50, 65, 90]), data.y)
print(cross3)
#将交叉报表归一化，利于分析数据
cross3_norm = cross3.div(cross3.sum(1).astype(float), axis = 0)
cross3_norm.plot(kind = "bar")
plt.show()


"""
相关系数：Correlation coefficient
相关系数：研究变量之间线性相关程度的量。
数值型变量
"""
import seaborn as sns

data_type = [] #创建一个空的list

for i in range(0, n-1):
    if np.dtype(data[Cnames[i]]) == 'int64':
        data_type.append(1)     # '1' 数值型 
    else:
        data_type.append(0)     #'0' 标称型

data_type = pd.Series(data_type)
data_bc = data[Cnames[data_type[data_type == 0].index]] # 标称型数据
data_num = data[Cnames[data_type[data_type == 1].index]] # 数值型数据

corr_mat = np.corrcoef(data_num.values.T)
"""
np.corrcoef(a)计算行与行之间的相关系数 
相关系数为1：表示两个随机变量完全线性正相关；
相关系数为−1：表示两个随机变量完全线性负相关。
"""

sns.set(font_scale = 1)
full_mat = sns.heatmap(corr_mat, cbar = True, annot = True, square = True,
                        annot_kws = {'size': 15}, 
                       yticklabels = data_num.columns,
                       xticklabels = data_num.columns)
"""
sns.heatmap()：热力图
利用热力图可以看数据表里多个特征两两的相似度。
"""
