import pandas as pd
from numpy import nan as NA

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
df = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)

m, n = df.shape #DataFrame的维数
# Missing value: Replace '?' with NA
df.replace('?', NA, inplace=True)

na_count = list(df.isnull().sum())
print('一共有%s个缺失值' % sum(na_count))

cols = []
for i in range(len(Cname)):
    if na_count[i] != 0 :
        cols.append(Cname[i])
print(', '.join(cols) + '特征存在缺失值')

na_sample = 0
for i in range(m):
    if df.loc[i, :].isnull().sum() != 0:
        na_sample += 1
print('有%s个样本存在缺失值'%(na_sample))


na_sample = 0
for i in range(m):
    if df.loc[i, :].isnull().sum() == 1:
        na_sample += 1
print('有%s个样本只有一个特征存在缺失值'%(na_sample))


na_sample = 0
for i in range(m):
    if df.loc[i, :].isnull().sum() == 2:
        na_sample += 1
print('有%s个样本只有两个特征存在缺失值'%(na_sample))

path = "C:\\Users\\lucky bill\\Desktop\\Personal\\Python\\Lecture1\\assignment.csv"
df.to_csv(path, na_rep = 'NA', index = False, header = True)
