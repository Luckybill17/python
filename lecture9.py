# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:53:46 2019

@author: Anmin Zou
"""

import pandas as pd
import numpy as np

#变量、引用和对象
#在Python中对每一个变量赋值时，就创建了一个指向等号右边的引用。
a = [4, 3, 2, 5]
#a为变量，[4, 3, 2, 5]为对象。
#定义新的变量：
b = a
c = b

print('\na =', a)
print('\nb =', b)
print('\nc =', c)

a.append(0)
print('\na =', a)
print('\nb =', b)
print('\nc =', c)

b.sort()
print('\na =', a)
print('\nb =', b)
print('\nc =', c)



path = "D:\\Work at STU\\Data\\Breast-Cancer-Data.txt"
data = pd.read_csv(path, header = None, engine = 'python')

Cnames = ['ID', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
          'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei',
          'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

"""
 #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10  肿块厚度
   3. Uniformity of Cell Size       1 - 10  细胞大小的均匀性
   4. Uniformity of Cell Shape      1 - 10  细胞形状的均匀性
   5. Marginal Adhesion             1 - 10  边缘粘
   6. Single Epithelial Cell Size   1 - 10  单上皮细胞的大小
   7. Bare Nuclei                   1 - 10  裸核
   8. Bland Chromatin               1 - 10  乏味染色体
   9. Normal Nucleoli               1 - 10  正常核
  10. Mitoses                       1 - 10  有丝分裂
  11. Class:                        (2 for benign, 4 for malignant) 良性的，恶性的
  
  Missing attribute values: 16 denoted by "?".  
  
  http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

"""

m, n = data.shape

data.columns = Cnames

y = data.Class

y.value_counts()

data = data.replace('?', np.NAN)
"""
data = data.replace('?', np.NAN) 变量data重新赋值
data.replace('?', np.NAN, inplace = True) ：变量data所指向的对象会改变
参数：inplace，默认值False。
如果是True，将会改变原对象。
见下例子
"""
s = pd.Series([0, 1, 2, 3, 4])
print(s)
t = s
s.replace(0, 5) #变量s所指向的对象并没有发生变化

print('If inplace = False, s =\n', s)
print('If inplace = False, t =\n', t)
s.replace(0, 5, inplace = True) # 变量s所指向的对象发生变化
print('If inplace = True, s =\n', s)
print('If inplace = True, t =\n', t)

E = data.isnull().sum(axis = 0)

"""
Bare_Nuclei（裸核） 存在缺失值
缺失值由出现频率最高的值代替
"""
data['Bare_Nuclei'].value_counts()

data = data.fillna(data['Bare_Nuclei'].value_counts().index[0])
"""
data = data.fillna(data['Bare_Nuclei'].value_counts().index[0])： 变量data重新赋值
data.fillna(data['Bare_Nuclei'].value_counts().index[0], inplace = True)：变量data所指向的对象会改变
参数：inplace，默认值False。
如果是True，将会改变原对象。
见下例子
"""
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                   columns=list('ABCD'))
df1 = df
df.fillna(0) #变量df所指向的对象并没有发生变化
df2 = df.fillna(0)
print('If inplace = False, df =\n', df)
print('If inplace = False, df1 =\n', df1)
print('df2 = \n', df2)
df.fillna(0, inplace = True) #变量df指向的对象发生变化
print('If inplace = True, df =\n', df)
print('If inplace = True, df1 =\n', df1)
print('df2 =\n ', df2)

data.loc[data.Class[data.Class == 4].index, 'Class'] = 1
data.loc[data.Class[data.Class == 2].index, 'Class'] = 0

x = data.drop(['ID', 'Class'], axis = 1)
y = data.Class

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, 
                                                    random_state = 25)

from sklearn.preprocessing import StandardScaler
#标准化
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
knn_fit = KNeighborsClassifier(n_neighbors = 3, p = 2, metric = 'minkowski')
knn_fit.fit(x_train, y_train)

"""
 n_neighbors : int, optional (default = 5)
 p : integer, optional (default = 2)
 |      Power parameter for the Minkowski metric.
 metric : string or callable, default 'minkowski'
 |      the distance metric to use for the tree.  The default metric is
 |      minkowski, and with p=2 is equivalent to the standard Euclidean
 |      metric. 
"""

y_train_pred = knn_fit.predict(x_train) #预测值
y_test_pred = knn_fit.predict(x_test) #预测值

def ComputeCM(y, y_pred):
    #用于计算TP，TN，FN，FP, ACC和混淆矩阵（两类问题：1和0）
    #y：真实值
    #y_pred：预测值
    TP = len(y_pred[y == 1][y_pred[y == 1] == 1])
    TN = len(y_pred[y == 0][y_pred[y == 0] == 0])
    FN = len(y_pred[y == 1][y_pred[y == 1] == 0])
    FP = len(y_pred[y == 0][y_pred[y == 0] == 1])
    ACC = (TP+TN)/len(y)
    Confusion_Matrix = pd.DataFrame([[TP, FN], [FP, TN]])
    Confusion_Matrix.columns = ['Predicted (y=1)', 'Predicted (y=0)']
    Confusion_Matrix.index = ['Actual (y=1)', 'Actual (y=0)']
    return TP, TN, FN, FP, ACC, Confusion_Matrix

TP1, TN1, FN1, FP1, ACC1, Confusion_Matrix1 = ComputeCM(y_train, y_train_pred)
print(Confusion_Matrix1)
TP2, TN2, FN2, FP2, ACC2, Confusion_Matrix2 = ComputeCM(y_test, y_test_pred)
print(Confusion_Matrix2)
    

train_Acc = len(y_train[y_train == y_train_pred])/len(y_train)
print('\nK-Nearest Neighbors - Train Accuracy:', np.round(train_Acc, 3))

test_Acc = len(y_test[y_test == y_test_pred])/len(y_test)
print('\nK-Nearest Neighbors - Test Accuracy:', np.round(test_Acc, 3))

k_set = [1, 2, 3, 4, 5, 6, 7, 8]
train_Acc_set = []
test_Acc_set = []

for k in k_set:
    knn_fit = KNeighborsClassifier(n_neighbors = k, p = 2, metric = 'minkowski')
    knn_fit.fit(x_train, y_train)
    y_train_pred = knn_fit.predict(x_train) #预测值
    y_test_pred = knn_fit.predict(x_test) #预测值
    train_Acc = len(y_train[y_train == y_train_pred])/len(y_train)
    test_Acc = len(y_test[y_test == y_test_pred])/len(y_test)
    train_Acc_set.append(np.round(train_Acc, 3))
    test_Acc_set.append(np.round(test_Acc, 3))
    
# Ploting accuracies over varied K-values
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.plot(k_set, train_Acc_set, 'k-', k_set, test_Acc_set, 'r-')
plt.axis([0.9, 8, 0.92, 1.005])
#plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])

for i in np.arange(len(k_set)):
    plt.text(k_set[i], train_Acc_set[i], str(train_Acc_set[i]), fontsize = 10)
    plt.text(k_set[i], test_Acc_set[i], str(test_Acc_set[i]), fontsize = 10)
    
"""
上述for循环等价于
for a, b in zip(k_set, train_Acc_set):
    plt.text(a, b, str(b), fontsize = 10)
for a, b in zip(k_set, test_Acc_set):
    plt.text(a, b, str(b),fontsize=10)
"""
    
plt.legend(('Training Set', 'Test Set'), loc = 'upper right')
plt.show()


"""
Recommendation Engines Example
"""
# DataFrame中的特殊索引符号loc和iloc。
#loc使用轴标签，iloc使用整数标签。
df = pd.DataFrame(np.arange(16).reshape((4, 4)), index = ['A', 'B', 'C', 'D'],
                  columns = ['One', 'Two', 'Three', 'Four'])

df.loc['A', ['One', 'Two']]
df.loc[:'C', :]
df.iloc[3, 3]
df.iloc[:3, [1, 3]]


path1 = "D:\\Work at STU\\Data\\ratings.csv"
ratings = pd.read_csv(path1)
print (ratings.head())
ratings.userId.value_counts()
ratings.movieId.value_counts()

path2 = "D:\\Work at STU\\Data\\movies.csv"
movies = pd.read_csv(path2)
print (movies.head())
movies.movieId.value_counts()


#Combining movie ratings & movie names
ratings = pd.merge(ratings[['userId', 'movieId', 'rating']], movies[['movieId', 'title']],
                   how = 'left', left_on = 'movieId', right_on = 'movieId')


rp = ratings.pivot_table(columns = ['movieId'], index = ['userId'], values = 'rating')
rp = rp.fillna(0)

rp.shape

# Converting pandas dataframe to numpy for faster execution in loops etc.
rp_mat = rp.as_matrix()

from scipy.spatial.distance import cosine

m, n = rp.shape

# User similarity matrix
mat_users = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            mat_users[i][j] = (1- cosine(rp_mat[i,:], rp_mat[j,:]))
        else:
            mat_users[i][j] = 0
            
pd_users = pd.DataFrame(mat_users, index = rp.index, columns = rp.index )


# Finding similar users
def topn_simusers(uid, n):
    users = pd_users.loc[uid,:].sort_values(ascending = False)
    topn_users = users.iloc[:n,]
    topn_users = topn_users.rename('score')    
    print ("Similar users as user:", uid)
    return pd.DataFrame(topn_users)

print (topn_simusers(uid = 17, n = 10))   

def topn_movieratings(uid, n_ratings):    
    uid_ratings = ratings.loc[ratings['userId'] == uid]
    uid_ratings = uid_ratings.sort_values(by = 'rating', ascending = [False])
    print ("Top", n_ratings, "movie ratings of user:", uid)
    return uid_ratings.iloc[:n_ratings,]    

print (topn_movieratings(uid = 596, n_ratings = 20))
