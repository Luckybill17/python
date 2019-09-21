# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 16:29:28 2019

@author: lucky bill
"""

#异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data')

#绘制箱线图
plt.boxplot(x = sunspots.Cname,
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','marksize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show()

# 计算上下四分位数
Q1 = sunspots.Cname.quantile(q = 0.25)
Q3 = sunspots.Cname.quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots.Cname[(sunspots.Cname > high_quantile) | (sunspots.Cname < low_quantile)]

print(value)
