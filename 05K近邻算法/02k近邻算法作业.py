from sklearn.datasets import load_iris,load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
def irisfl():
    '''

    :return:
    '''
    li = load_iris()
    # 取出数据当中的特征值和目标值
    y = li.target
    x = li.data
    # 进行数据的分割训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程(标准化)
    std = StandardScaler()
    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    # 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=5)
    # fit，predict，score
    knn.fit(x_train, y_train)
    # 得出预测结果
    y_predict = knn.predict(x_test)
    print("预测的目标签到位置：", y_predict)
    # 得出准确率
    print("预测的准确值：", knn.score(x_test, y_test))

if __name__ == '__main__':
    irisfl()
