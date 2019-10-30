# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:57:54 2019

@author: Anmin Zou
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

H = []
Gini = []
x = np.arange(0, 1.01, 0.01)
for i in x:
    p = i
    Gini.append(1-p**2-(1-p)**2)
    if (p == 0) or (p == 1):
        H.append(0)
    else: 
       Entropy = -p*np.log2(p)-(1-p)*np.log2(1-p)
       H.append(Entropy)
 
plt.plot(x, H, 'k-', x, Gini, 'r-')
plt.xlabel('The probability p')
plt.legend(('Entropy', 'Gini'))
plt.show()


path = "C:\\Users\\lucky bill\\Desktop\\Personal\\Python\\Lecture7\\WA_Fn-UseC_-HR-Employee-Attrition.csv"

data = pd.read_csv(path)

"""
数据集：IBM Watson的HR员工流失
https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset 
https://www.ibm.com/communities/analytics/watson-analytics-blog/hr-employee-attrition/ 
根据IBM声明，“这是由IBM数据科学家创建的虚构数据集”。
数据集包括1470名员工（行）和35个特征（列），一部分(237)已经离职（Attrition = “Yes”）。
EmployeeCount、EmployeeNumber、Over18、StandardHours四个特征与建模为无关，
故删除这四个特征。
类别型变量：
BusinessTravel，Department，EducationField，Gender，JobRole
MaritalStatus，OverTime
"""

"""
删除EmployeeCount、EmployeeNumber、Over18、StandardHours
"""

print(data.EmployeeCount)
print(data.EmployeeNumber)
print(data.Over18)
print(data.StandardHours)

drop_v =['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
data = data.drop(drop_v, axis = 1)

print(data.head())
print(data.columns)
data.describe()
data.shape

print(data.Attrition.value_counts())

fig = plt.figure(figsize = (10, 10))
y = ["No", "Yes"]
ax = sns.categorical.barplot(y, np.array(data.Attrition.value_counts(normalize = True)), 
                             saturation = 1)
ax.set_xticklabels(y)
ax.set_title("Attrition")
ax.set_xlabel("")
ax.set_ylabel("Frequency")
ax.set_ylim([0, 1])
plt.show()


con_columns = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
'HourlyRate', 'JobInvolvement', 'JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
'YearsWithCurrManager'] #数值变量

cat_columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
               'JobRole', 'MaritalStatus', 'OverTime'] #类别变量
"""
类别型变量
"""
print(data.BusinessTravel.value_counts())
print(data.Department.value_counts())
print(data.EducationField.value_counts())
print(data.Gender.value_counts())
print(data.JobRole.value_counts())
print(data.MaritalStatus.value_counts())
print(data.OverTime.value_counts())

num_data = data[con_columns]
cat_data = data[cat_columns]

data.loc[data.Attrition[data.Attrition == 'Yes'].index, 'Attrition'] = 1
data.loc[data.Attrition[data.Attrition == 'No'].index, 'Attrition'] = 0


"""
data.loc[data.Attrition[data.Attrition == 'Yes'].index, 'Attrition'] = 1; data.Attrition.dtype 为 int
data.Attrition[data.Attrition == 'Yes'] = 1
data.Attrition[data.Attrition == 'No'] = 0
data.Attrition
data.Attrition.dtype 为 object
"""


y = data.Attrition

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

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC_fit = GBC.fit(x_train,y_train)

print ("\nGradient Boosting Classifier - Train Confusion Matrix\n\n", pd.crosstab(y_train, GBC.predict(x_train),
                                            rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nGradient Boosting Classifier - Train accuracy:", round(accuracy_score(y_train, GBC.predict(x_train)),3))
print ("\nGradient Boosting Classifier - Train Classification Report\n", 
       classification_report(y_train, GBC.predict(x_train)))

print ("\n\nGradient Boosting Classifier - Test Confusion Matrix\n\n",pd.crosstab(y_test, GBC.predict(x_test),
                                            rownames = ["Actuall"], colnames = ["Predicted"]))      
print ("\nGradient Boosting Classifier - Test accuracy:",round(accuracy_score(y_test, GBC.predict(x_test)),3))
print ("\nGradient Boosting Classifier - Test Classification Report\n", classification_report(y_test, GBC.predict(x_test)))