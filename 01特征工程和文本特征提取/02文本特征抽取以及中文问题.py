from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
def countvec():
    '''
    对文本进行特征值化
    :return:
    '''
    cv = CountVectorizer()
    data = cv.fit_transform(['life is short,i like python','life is too long,i dislike python'])
    print(cv.get_feature_names())
    print(data.toarray())
    jieba_cut = jieba.cut("人生苦短，我用pyhton")
    print(jieba_cut)
    return None

def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人都不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正的涵义的秘密取决于如何将其与我们所了解的事务相联系。")
    #转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    print(content1)
    #列表转为字符串
    c1 = ' ' .join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3

def hanzivec():
    '''
    中文特征值化
    1,今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人都不要放弃今天。
    2，我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。
    3，如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正的涵义的秘密取决于如何将其与我们所了解的事务相联系。
    :return:
    '''
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())

    return None


def tfidvec():
    '''
    重要性
    :return:
    '''

    c1,c2,c3 = cutword()
    print(c1,c2,c3)
    tfid = TfidfVectorizer()
    data = tfid.fit_transform([c1,c2,c3])
    print(tfid.get_feature_names())
    print(data.toarray())

    return None


if __name__ == '__main__':
    tfidvec()
