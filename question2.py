#数据中数值型数据 age 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['age'].quantile(q = 0.25)
Q3 = sunspots['age'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['age'][(sunspots['age'] > high_quantile) | (sunspots['age'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots.age,
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到age 型数据存在异常值

#数据中数值型数据 fnlwgt 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['fnlwgt'].quantile(q = 0.25)
Q3 = sunspots['fnlwgt'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['fnlwgt'][(sunspots['fnlwgt'] > high_quantile) | (sunspots['fnlwgt'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots.fnlwgt,
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到fnlwgt 型数据存在异常值

#数据中数值型数据 edu-num 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['edu-num'].quantile(q = 0.25)
Q3 = sunspots['edu-num'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['edu-num'][(sunspots['edu-num'] > high_quantile) | (sunspots['edu-num'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots['edu-num'],
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到edu-num 型数据存在异常值

#数据中数值型数据 cap-g 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['cap-g'].quantile(q = 0.25)
Q3 = sunspots['cap-g'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['cap-g'][(sunspots['cap-g'] > high_quantile) | (sunspots['cap-g'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots['cap-g'],
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到cap-g 型数据存在异常值

#数据中数值型数据 cap-g 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['cap-l'].quantile(q = 0.25)
Q3 = sunspots['cap-l'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['cap-l'][(sunspots['cap-l'] > high_quantile) | (sunspots['cap-l'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots['cap-l'],
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到cap-l 型数据存在异常值

#数据中数值型数据 cap-g 异常值检测
import pandas as pd
import matplotlib.pyplot as plt

Cname = ['age', 'wc', 'fnlwgt', 'edu', 'edu-num', 
         'ms', 'occ', 'rs', 'race', 'sex', 
         'cap-g', 'cap-l', 'hpw', 'native', 'y'] #Name of columns
sunspots = pd.read_csv('./adult.data', sep = ', ', header = None,  names = Cname)


# 计算上下四分位数
Q1 = sunspots['hpw'].quantile(q = 0.25)
Q3 = sunspots['hpw'].quantile(q = 0.75)

#异常值判断标准， 1.5倍的四分位差 计算上下须对应的值
low_quantile = Q1 - 1.5*(Q3-Q1)
high_quantile = Q3 + 1.5*(Q3-Q1)

# 输出异常值
value = sunspots['hpw'][(sunspots['hpw'] > high_quantile) | (sunspots['hpw'] < low_quantile)]
print (value)

#绘制箱线图
plt.boxplot(x = sunspots['hpw'],
            whis = 1.5,#1.5倍的四分位差
            widths = 0.7,
            patch_artist = True,#填充箱体颜色
            showmeans= True,# 显示均值
            boxprops = {'facecolor':"blue"},#箱体填充色
            flierprops = {'markerfacecolor':'red','markeredgecolor':'red','markersize':4},#异常点填充色、边框、大小
            meanprops = {'marker':'D','markerfacecolor':'black','markersize':4},#均值点标记符号、填充色、大小
            medianprops = {'linestyle':'--','color':'orange'},#中位数处的标记符号、填充色、大小
            labels = [''])
plt.show() 
#运行后我们可以看到hpw 型数据存在异常值