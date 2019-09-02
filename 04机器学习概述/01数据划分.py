from sklearn.datasets import load_iris,load_boston
from sklearn.model_selection import train_test_split
li = load_iris()
'''
print("获取特征值")
print(li.data)

print("获取目标值")
print(li.target)
print("描述")
print(li.DESCR)
#注意返回值，训练集train   x_train,y_train      测试集  test x_test,y_test
x_train, x_test, y_train, y_test = train_test_split(li.data,li.target,test_size=0.25)
print("训练集特征值和目标值：",x_train,y_train)
print("测试集特征值和目标值：",x_test,y_test)
'''
lb = load_boston()
print(lb.data)
print(lb.target)
print(lb.DESCR)