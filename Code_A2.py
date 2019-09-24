# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:46:34 2019

@author: Anmin Zou
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 


"""
用蒙特卡罗方法来验证中心极限定理
"""

n = 1000
N = 1000
a = 0
b = 4
mu = (a+b)/2
sigma = np.sqrt((b-a)**2/12)
sam_m = []

np.random.seed(123)
for _ in range(0, N):
    sample = np.round(np.random.uniform(a, b, n), 4)
    smean = np.round((sum(sample)-n*mu)/(sigma*np.sqrt(n)), 4)
    sam_m.append(smean)

plt.hist(sample, 50, density = True) 
plt.show()
   
plt.hist(sam_m, 50, density = True) 
plt.show()

"""
Q-Q plot
"""
sam_m.sort()
prob = (np.arange(N)+1/2)/N
q = np.round(norm.ppf(prob, 0, 1), 4) 
plt.scatter(x = q, y = sam_m, color = 'red')
# Add a 45-degree reference line
plt.plot([sam_m[0], sam_m[N-1]], [sam_m[0], sam_m[N-1]], color = 'blue', 
         linewidth = 2)
plt.xlabel('Theoretical Quantiles (q)')
plt.ylabel('Ordered Observed Values (y)' )
plt.title('Normal Q-Q plot')
plt.show()
