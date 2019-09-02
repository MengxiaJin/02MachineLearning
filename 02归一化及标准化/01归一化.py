from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,Imputer
import numpy as np

def mm():
    '''
    归一化处理
    :return:
    '''
    mm = MinMaxScaler(feature_range=(2,3))
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    #print(mm.get_params())
    print(data)
    return None

def stand():
    '''
    标准化缩放
    :return:
    '''
    sta = StandardScaler()
    data = sta.fit_transform([[1,-1,3],[2,4,2],[4,6,-1]])
    print(data)
    return None

def im():
    '''
    缺失值处理
    :return:
    '''
    im = Imputer(missing_values='NaN',strategy='mean',axis=0)
    data = im.fit_transform([[1.,2.],[np.nan,3.],[7.,6.]])
    print(data)
    return None

if __name__ == '__main__':
    im()