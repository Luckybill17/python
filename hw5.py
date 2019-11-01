# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:23:35 2019

@author: lucky bill
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

Cname = [
    'age', 'wc', 'fnlwgt', 'edu', 'edu-num', 'ms', 'occ', 'rs', 'race', 'sex',
    'cap-g', 'cap-l', 'hpw', 'native', 'y'
]

df = pd.read_csv('./adult.data',
                 sep=', ',
                 header=None,
                 names=Cname,
                 engine='python')

print(df.head())
print(df.columns)
df.describe()
df.shape
print(df.info())

# 缺失值统计
df.replace('?', np.nan, inplace=True)
df.isnull().sum()[df.isnull().sum()>0]

# 删除缺失值
df.dropna(axis=0, inplace=True)
df.shape

    
fig = plt.figure(figsize = (10, 10))
y = ["<=50K", ">50K"]
ax = sns.categorical.barplot(y, np.array(df.y.value_counts(normalize = True)), 
                             saturation = 1)
ax.set_xticklabels(y)
ax.set_title("y")
ax.set_xlabel("")
ax.set_ylabel("frequency")
ax.set_ylim([0, 1])
plt.show()

con_columns = ['age','fnlwgt','edu-num','cap-g','cap-l','hpw'] #数值变量

cat_columns = [ 'wc', 'edu', 'ms', 'occ', 'rs', 'race','sex', 'native', 'y'] #类别变量

print(df.wc.value_counts())
print(df.edu.value_counts())
print(df.ms.value_counts())
print(df.occ.value_counts())
print(df.rs.value_counts())
print(df.race.value_counts())
print(df.sex.value_counts())
print(df.native.value_counts())
print(df.y.value_counts())

num_data = df[con_columns]
cat_data = df[cat_columns]

df.loc[df.y[df.y == '>50K'].index, 'y'] = 1
df.loc[df.y[df.y == '<=50K'].index, 'y'] = 0

y = df.y

dummy_cat_data = pd.get_dummies(cat_data)
#将类别型变量转换为哑铃变量。

X = pd.concat([num_data, dummy_cat_data], axis = 1)
type(X)
type(y)
"""
X：DataFrame
y: Series

model = sm.Logit.from_formula(formula, data = train_data)
re = model1.fit(method = 'ncg')
train_data： DataFrame， 类别型变量不需要哑铃化处理
"""

X = np.array(X)
y = np.array(y)

"""
将数据随机分为训练子集（Training Set）和测试子集（Test Set）
70%：30% 或 80%：20%
"""
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, 
                                                    random_state = 42)

from sklearn.preprocessing import StandardScaler
#数据的标准化处理
"""
 StandardScaler(copy = True, with_mean = True, with_std = True)
 |  
 |  Standardize features by removing the mean and scaling to unit variance
 |  
 |  The standard score of a sample `x` is calculated as:
 |  
 |      z = (x - u) / s
 |  
 |  where `u` is the mean of the training samples or zero if `with_mean = False`,
 |  and `s` is the standard deviation of the training samples or one if
 |  `with_std = False`.
"""
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
                      
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion = "gini", max_depth = 5,
         min_samples_split = 2, min_samples_leaf = 1, random_state = 42)
DTC_fit = DTC.fit(x_train, y_train)
"""
fit(X, y)
X: array-like or sparse matrix, shape = [n_samples, n_features]
y: array-like, shape = [n_samples] or [n_samples, n_outputs]
因此，类别型变量需要哑铃化
"""

print ("\nDecision Tree - Train Confusion Matrix\n\n", pd.crosstab(y_train, DTC.predict(x_train),
                                            rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nDecision Tree - Train accuracy:", round(accuracy_score(y_train, DTC.predict(x_train)),3))
print ("\nDecision Tree - Train Classification Report\n", 
       classification_report(y_train, DTC.predict(x_train)))

print ("\n\nDecision Tree - Test Confusion Matrix\n\n",pd.crosstab(y_test, DTC.predict(x_test),
                                            rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nDecision Tree - Test accuracy:",round(accuracy_score(y_test, DTC.predict(x_test)),3))
print ("\nDecision Tree - Test Classification Report\n", classification_report(y_test, DTC.predict(x_test)))


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_fit = RandomForestClassifier(n_estimators = 5000, criterion = "gini",
                                max_depth = 5, min_samples_split = 2, bootstrap = True,
                                max_features = 'auto', random_state = 42, 
                                min_samples_leaf = 1, class_weight = {0:0.3, 1:0.7})
#rf_fit = RandomForestClassifier()
rf_fit.fit(x_train,y_train)       

print ("\nRandom Forest - Train Confusion Matrix\n\n", pd.crosstab(y_train, rf_fit.predict(x_train),
                                rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nRandom Forest - Train accuracy", round(accuracy_score(y_train, rf_fit.predict(x_train)), 3))
print ("\nRandom Forest  - Train Classification Report\n", classification_report(y_train, rf_fit.predict(x_train)))

print ("\n\nRandom Forest - Test Confusion Matrix\n\n", pd.crosstab(y_test, rf_fit.predict(x_test),
                                                rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nRandom Forest - Test accuracy",round(accuracy_score(y_test, rf_fit.predict(x_test)), 3))
print ("\nRandom Forest - Test Classification Report\n", classification_report(y_test, rf_fit.predict(x_test)))
