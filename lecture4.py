# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:45:35 2019

@author: Anmin Zou
"""

import pandas as pd
import numpy as np
from numpy import nan as NA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
定义Sigmoid函数
"""

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

y = []
x = np.arange(-10, 10, 0.01)
for i in x:
    y.append(sigmoid(i))
 
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.show()

"""
数据集：Adult
方法：逻辑回归
"""
"""
读取已经处理过的的数据
"""

path = "D:\\Work at STU\\Data\\adult1.csv"
data = pd.read_csv(path)


"""
重新定义column names
"""
data.columns = ['age', 'wc', 'fnlwgt', 'edu', 'edu_num', 'ms',  'occ', 
               'rs', 'race', 'sex', 'cap_g', 'cap_l', 'hpw', 'native', 'y']

m, n = data.shape

data = data.drop('fnlwgt', axis = 1)
#删除 'fnlwgt'

Num_name = ['age',   'edu_num',     
         'cap_g', 'cap_l', 'hpw'] #数值变量 

Class_name = ['wc', 'edu', 'ms', 'occ', 'rs',
           'race', 'sex', 'native'] #类别变量 


data.wc.value_counts()
data.edu.value_counts()
data.ms.value_counts()
data.occ.value_counts()
data.rs.value_counts()
data.race.value_counts()
data.native.value_counts()

w1 = data.y[data.wc == 'Without-pay']
print(len(w1[w1==1])/len(w1))
w2 = data.y[data.wc == 'Self-emp-inc']
print(len(w2[w2==1])/len(w2))

w3 = data.y[data.wc == 'Federal-gov']
print(len(w3[w3==1])/len(w3))

data.edu[data.edu == 'Preschool'] = 'Droupouts'
data.edu[data.edu == '1st-4th'] = 'Droupouts'
data.edu[data.edu == '5th-6th'] = 'Droupouts'
data.edu[data.edu == '7th-8th'] = 'Droupouts'
data.edu[data.edu == '9th'] = 'Droupouts'
data.edu[data.edu == '10th'] = 'Droupouts'
data.edu[data.edu == '11th'] = 'Droupouts'
data.edu[data.edu == '12th'] = 'Droupouts'
data.edu.value_counts()

data.ms[data.ms == 'Divorced'] = 'Separated'
data.ms[data.ms == 'Married-spouse-absent'] = 'Separated'
data.ms[data.ms == 'Widowed'] = 'As-Widowed'
data.ms[data.ms == 'Married-civ-spouse'] = 'Married'
data.ms[data.ms == 'Married-AF-spouse'] = 'Married'
data.ms.value_counts()

data.native[data.native != 'United-States'] = 'Not-USA'
data.native[data.native == 'United-States'] = 'USA'
data.native.value_counts()

"""
data.occ[data.occ == 'Sales'] = 'Services'
data.occ[data.occ == 'Other-service'] = 'Services'
data.occ[data.occ == 'Protective-serv'] = 'Services'
data.occ[data.occ == 'Priv-house-serv'] = 'Services'
data.occ[data.occ == 'Handlers-cleaners'] = 'Services'
data.occ[data.occ == 'Armed-Forces'] = 'Services'
data.occ[data.occ == 'Craft-repair'] = 'Tech'
data.occ[data.occ == 'Machine-op-inspct'] = 'Tech'
data.occ[data.occ == 'Tech-support'] = 'Tech'
data.occ[data.occ == 'Farming-fishing'] = 'Tech'
data.occ.value_counts()
"""
s1 = data.y[data.occ=='Farming-fishing']
print(len(s1[s1==1])/len(s1))
s2 = data.y[data.occ=='Priv-house-serv']
print(len(s2[s2==1])/len(s2))
s3 = data.y[data.occ=='Other-service']
print(len(s3[s3==1])/len(s3))
s4 = data.y[data.occ=='Handlers-cleaners']
print(len(s4[s4==1])/len(s4))
data.occ[data.occ == 'Other-service'] = 'Services'
data.occ[data.occ == 'Priv-house-serv'] = 'Services'
data.occ[data.occ == 'Handlers-cleaners'] = 'Services'
data.occ[data.occ == 'Armed-Forces'] = 'Services'

r1 = data.y[data.race == 'Other']
print(len(r1[r1==1])/len(r1))
r2 = data.y[data.race == 'Amer-Indian-Eskimo']
print(len(r2[r2==1])/len(r2))
r3 = data.y[data.race == 'Black']
print(len(r3[r3==1])/len(r3))
data.race[data.race == 'Amer-Indian-Eskimo'] = 'As-Other'
data.race[data.race == 'Black'] = 'As-Other'
data.race[data.race == 'Other'] = 'As-Other'

rs1 = data.y[data.rs == 'Wife']
print(len(rs1[rs1==1])/len(rs1))
rs2 = data.y[data.rs == 'Husband']
print(len(rs2[rs2==1])/len(rs2))

rs3 = data.y[data.rs == 'Not-in-family']
print(len(rs3[rs3==1])/len(rs3))
rs4 = data.y[data.rs == 'Unmarried']
print(len(rs4[rs4==1])/len(rs4))

rs5 = data.y[data.rs == 'Own-child']
print(len(rs5[rs5==1])/len(rs5))
rs6 = data.y[data.rs == 'Other-relative']
print(len(rs6[rs6==1])/len(rs6))

data.rs[data.rs == 'Not-in-family'] = 'As-Not-in-family'

"""
将数据随机分为训练子集（Training Set）和测试子集（Test Set）
70%：30% 或 80%：20%
"""
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, train_size = 0.7, 
                                         random_state = 42)
"""
train_test_split(): 将数组或矩阵随机分为训练和测试子集 
random_state = int：设置随机数种子
"""

"""
归一化处理：数值型数据，-1和1之间
"""

"""
train_max = []
for i in Num_name:
    max1 = max(abs(train_data[i]))
    train_max.append(max1)
    train_data.loc[:, i] /= max1 
    
train_max = pd.Series(train_max, index = Num_name)
"""

"""
标准化处理：数值型数据
"""

train_mean = []
train_std = []
for i in Num_name:
    mean_t = np.mean(train_data[i])
    std_t = np.std(train_data[i])
    train_mean.append(mean_t)
    train_std.append(std_t)
    train_data.loc[:, i] = (train_data.loc[:, i]-mean_t)/std_t 

train_mean = pd.Series(train_mean, index = Num_name)
train_std = pd.Series(train_std, index = Num_name)


"""
搭建模型：应用第三方库statsmodels
"""
import statsmodels.api as sm
formula1 = "y ~ age+edu_num+cap_g+cap_l+hpw+wc+ms+native+edu+rs+occ+race+sex"
model1 = sm.Logit.from_formula(formula1, data = train_data)
re1 = model1.fit(method = 'ncg')
"""
'ncg' for Newton-conjugate gradient
共轭梯度法（Conjugate Gradient）是介于最速下降法与牛顿法之间的一个方法，
它仅需利用一阶导数信息，但克服了最速下降法收敛慢的缺点，
又避免了牛顿法需要存储和计算Hesse矩阵并求逆的缺点。
"""
re1.summary()
print("AIC = ", np.round(re1.aic, 2))
print("BIC = ", np.round(re1.bic, 2))

"""
Logit Marginal Effects
"""
re1.get_margeff(at = "overall").summary()

"""
formula2 = "y ~ age+edu_num+cap_g+cap_l+hpw+wc+ms+rs+occ+race+sex"
model2 = sm.Logit.from_formula(formula2, data = train_data)
re2 = model2.fit(method = 'ncg')
re2.summary()
print("AIC = ", np.round(re2.aic, 2))
print("BIC = ", np.round(re2.bic, 2))
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

"""
测试集数据的归一化处理
"""

"""
for i in Num_name:
    test_data.loc[:, i] /= train_max[i] 
"""

"""
测试集数据的标准化处理
"""

for i in Num_name:
    test_data.loc[:, i] = (test_data.loc[:, i]
    -train_mean[i])/train_std[i] 
  
y_test = test_data.y

y_pred = re1.predict(test_data)
cutoff = 0.5
y_pred[y_pred >= cutoff] = 1
y_pred[y_pred < cutoff] = 0
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

predicted_values = re1.predict(test_data)
"""
re1.predict():预测概率值
"""
fpr, tpr, thresholds = roc_curve(y_test, predicted_values)
roc_auc = auc(fpr, tpr)

lw = 2
plt.plot(fpr, tpr, color = 'darkorange',
         lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc) 
###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc = "lower right")
plt.show()

accuracy = (tn+tp)/(tn+fp+fn+tp)
print("Accuracy = ", np.round(accuracy, 2))

TPR = tp/(tp+fn)
TNR = tn/(tn+fp)
print("TPR = ", np.round(TPR, 2))
print("TNR = ", np.round(TNR, 2))

