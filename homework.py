from random import randint
num = eval(input("请输入要测试的次数："))   #num次数越大结果越接近真实值

a1,a2,a3 = 0,0,0           #a1，a2，a3分别为羊1羊2和车
##如果参赛者不改变选择
for i in range(1,num+1):
    people = randint(1,3)  			#模拟参赛者随机选择一个门，并循环num次
    if people == 3:
        a3 = a3 + 1              #每次选择的门进行计数
    elif people == 1:
        a1 = a1 + 1
    else:
        a2 = a2 + 1

print("不更改选择时选中车的概率为：{}".format(a3/num))
print("不更改选择时选中羊1号的概率为：{}".format(a1/num))
print("不更改选择时选中羊2号的概率为：{}".format(a2/num))
print("\n")

a1,a2,a3 = 0,0,0 

##如果参赛者改变选择
for i in range(1,num+1):
    people = randint(1,3)
    if people == 1:				#如果第一次选择羊1号，羊2号会被主持人打开，参赛者改选车
        a3 = a3 + 1
    elif people == 2:			#第一次选择羊2号，参赛者同样会改为车
        a3 = a3 + 1
    else:					#如果第一次选择了车，参赛者会改为羊1号或羊2号
        if randint(1,2) == 1:
            a1 = a1 + 1
        else :
            a2 = a2 + 1

print("更改选择时选中车的概率为：{}".format(a3/num))
print("更改选择时选中羊1号的概率为：{}".format(a1/num))
print("更改选择时选中羊2号的概率为：{}".format(a2/num))
print("\n")