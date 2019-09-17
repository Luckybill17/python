# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:01:51 2019

@author: Anmin Zou
"""

import pandas as pd
import numpy as np
from numpy import nan as NA
from scipy import stats


"""
创建一个数组
"""
data = np.array([4, 5, 1, 2, 7, 2, 6, 9, 3])

"""
计算均值（Mean）
"""
dt_mean = np.mean(data)
print('Mean:', np.round(dt_mean, 2))

"""
计算中位数（Median）
"""
dt_median = np.median(data)
print('Median:', dt_median)

"""
计算众数（Mode）
"""
dt_mode = stats.mode(data)
print('Mode:', dt_mode[0][0])

"""
np.array()：创建一个多维数组；
np.zeros()：一次性创建全0数组；
np.ones()：一次性创建全1数组；
np.arange():
"""

np.zeros(10)
np.ones((3, 6))
np.arange(15)

"""
至少二人生日相同问题
"""
def two_DOB(n):
    """
    计算n个人中至少有二人生日相同的概率
    参数： n, int 
    输出： prob, 至少有二人生日相同的概率
    """
    if n >= 365:
        prob = 1
    else:
        a = 1
        for j in range(0, n):
            a *= 365-j
        prob = 1-a/(365**n)
    return prob

n = [5, 10, 20, 50, 100]
prob = [] #定义一个空的list
for i in range(0, len(n)):
    prob.append(two_DOB(n[i]))
    
prob = np.round(prob, 2)
print ('Probability of at least two people having the same birthday:', 
       prob)
    
"""
a *= b:  a = a*b
a += b:  a = a+b
a -= b:  a = a-b
a /= b:  a = a/b
a ** n:  a的n次幂
"""

"""
数据结构：列表（list）
列表的长度是可以变的，它所包含的内容也是可以修改的。
可以用中括号[ ]或者list函数来定义列表。

"""

a_list = [] #创建一个空的列表
print('a_list =', a_list)
b_list = [1, 2, 3, 'a']
print('b_list =', b_list)
c_list = ['a', 'b', 'c']
print('c_list =', c_list)

"""
增加或移除元素
使用append方法可以将元素添加到列表的尾部。
使用insert方法可以将元素插入到指定的列表位置。插入位置的范围在0到列表长度之间。
使用pop方法可以将指定位置的元素移除并返回。
"""
a_list.append(1)
print('a_list =', a_list)
a_list.append('abc')
print('a_list =', a_list)

b_list.insert(2, 'one')
print('b_list =', b_list)

c_list.pop(1)
print('c_list =', c_list)

"""
数据结构：字典（dict）
字典是拥有灵活尺寸的键值对集合，其中键和值都是Python对象。
用大括号{ }是创建字典的一种方式，在字典中用逗号将键值对分离。
"""
empty_dict = {} #创建一个空的字典
dict1 = {'a': 1, 'b': 'Monday', 'c': [1, 2, 3]} 

"""
可以访问、插入或设置字典中的元素
可以使用del关键字或pop方法删除值；pop方法会在删除的同时返回被删的值，并删除键。
"""

print(dict1['b']) 
dict1[5] = [3, 7] 
del dict1[5]

ret = dict1.pop('c')
print('ret = ', dict1)


"""
数据结构：集合（set）
集合是一种无序且元素唯一的容器。可以认为集合是一个只有键没有值的字典。
集合有两种创建方式：通过set函数或者用大括号{ }。
集合支持数学上的集合操作，例如并集、交集、差集。
"""
set1 = set([1, 2, 3, 2, 5, 3, 4])
print('set1 = ', set1)

set2 = {2, 2, 3, 4, 4, 6}
print('set2 = ', set2)

"""
集合操作；
交集（set1&set2, set1.intersection(set2)）
并集 (set1|set2, set1.union(set2))
差集 (set1-set2, set1.difference(set2))：在set1不在set2中的元素
"""
set1&set2
set1|set2
set1-set2

"""
蒙特·卡罗方法的应用:
Q-Q plot 验证数据是否服从某一给定的分布
X~Norm(mu, sigma)，验证Y=(X-mu)/sigma~Norm(0, 1)
"""

from scipy.stats import norm
import matplotlib.pyplot as plt 
q = norm.cdf(3, 0, 1) # Cumulative distribution function evaluated at `x'.
norm.ppf(q) # Return a quantile corresponding to the lower tail probability q.

n = 10000
mu = 1
sigma = 2
np.random.seed(123)
x = np.random.normal(mu, sigma, n)
"""
np.random.normal()： 正态分布
np.random.uniform()： 均匀分布
np.random.binomial()： 二项分布
np.random.exponential()： 指数分布
np.random.poisson()： 泊松分分布
"""

y = (x-mu)/sigma
y.sort()
prob = (np.arange(n)+1/2)/n
q = norm.ppf(prob, 0, 1) 

   
count, bins, ignored = plt.hist(y, 300, density = True) #直方图
plt.show()
"""
直方图(Histogram)，又称质量分布图，是一种统计报告图，
由一系列高度不等的纵向条纹或线段表示数据分布的情况。 
一般用横轴表示数据类型，纵轴表示分布情况。

直方图可以被归一化以显示“相对”频率。 
显示属于几个类别中的每个类别的比例，其高度等于1。
"""

"""
Q-Q plot: Method 1
"""
plt.scatter(x = q, y = y, color = 'red')
# Add a 45-degree reference line
plt.plot([y[0], y[n-1]], [y[0], y[n-1]], color = 'blue', linewidth = 2)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values' )
plt.title('Normal Q-Q plot')
plt.show()

"""
Q-Q plot: Method 2
"""
stats.probplot(y, dist = "norm", plot = plt)
plt.title("Normal Q-Q plot")
plt.show()