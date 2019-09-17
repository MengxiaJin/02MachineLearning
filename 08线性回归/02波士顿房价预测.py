from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
def mylinear():
    '''
    线性回归预测房价
    :return:
    '''

    #获取数据
    lb = load_boston()
    #分割数据集到训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(lb.data,lb.target,test_size=0.25)
    #进行标准化
    #特征值和目标值是都必须进行标准化处理，实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    #estimator预测
    #正规方程求解方式预测结果
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    print(lr.coef_)

    #预测测试集的房子价格
    y_lr_preidct = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程测试集里面每个房子的价格：",y_lr_preidct)
    print("分数：",lr.score(x_test,y_test))
    print("正规方程的均方误差：",mean_squared_error(std_y.inverse_transform(y_test),y_lr_preidct))

    # 梯度下降求解方式预测结果
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)

    # 预测测试集的房子价格
    y_sgd_preidct = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降测试集里面每个房子的价格：", y_sgd_preidct)
    print("分数：", sgd.score(x_test, y_test))
    print("梯度下降的均方误差：",mean_squared_error(std_y.inverse_transform(y_test),y_sgd_preidct))

    # 岭回归求解方式预测结果
    rd = Ridge()
    rd.fit(x_train, y_train)
    print(rd.coef_)

    # 预测测试集的房子价格
    y_rd_preidct = std_y.inverse_transform(rd.predict(x_test))
    print("梯度下降测试集里面每个房子的价格：", y_rd_preidct)
    print("分数：", rd.score(x_test, y_test))
    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_preidct))




if __name__ == '__main__':
    mylinear()