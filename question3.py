import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from numpy import nan as NA
from scipy import stats

#蒙特卡罗方法随机生成数字并取平均值
samples_mean = [] #平均值的数组
for _ in range(10000):  #这个是范围（对取到的10000个整数操作）
    sample = np.random.uniform(-4,4,10000) #随机生成10000个范围在【-4,4】符合均匀分布的数字 
    samples_mean.append(sample.mean())
samples_mean=np.array( samples_mean) 
plt.hist(samples_mean,bins=30,color='g')#构建直方图bins为横坐标取30个值
plt.grid()
plt.show()

#构建Q-Qplot图验证中心极限定理
n = 10000
mu = 0
sigma = 4*pow(3,0.5)/3  # 概率分布均值为0，标准差为4√3/3，输出10000个值

y = (samples_mean-mu)*pow(n,0.5)/sigma #中心极限定理公式
y.sort()
prob = (np.arange(n)+1/2)/n
q = norm.ppf(prob, 0, 1)

# Q-Q plot method 1
plt.scatter(x = q, y = y, color = 'red')
# Add a 45-degree reference line
plt.plot([y[0], y[n-1]], [y[0], y[n-1]], color = 'blue', linewidth = 2)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values' )
plt.title('Normal Q-Q plot')
plt.show()
