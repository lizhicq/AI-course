# _*_ coding:utf-8 _*_
import gensim
import pandas as pd

dim = 200

def read_data():
    print("正在准备数据...")
    data = []
    data_in =  open('set_0204.txt', 'r',encoding='utf8')
    r = data_in.readlines()
    for line in r:
        line_list = line.strip().split('\t')
        if len(line_list)==2:
            feas = line_list[1]
            data.append(feas.split(":"))
    return data


sentences = read_data()
# print('sentences is: ',sentences)
# sentences = [['first', 'sentence'], ['second', 'sentence','is']]
print("正在进行模型训练...")
model = gensim.models.Word2Vec(sentences, min_count=1, size=dim)
print("模型训练完毕...")
subject='gzsx'
savepath = "word2vec_{1}.model".format(subject, dim)
model.save(savepath)
print("保存模型完毕！")


savepath = "word2vec_200.model"


#测试
print('-----------------------------------------')
model = gensim.models.Word2Vec.load(savepath)
print('和直线相近似的词有：')
print(pd.Series(model.most_similar(u'直线',topn = 10)))
print('和参数方程相近似的词有：')
print(pd.Series(model.most_similar(u'参数方程',topn = 10)))
print('和余弦函数相近似的词有：')
print(pd.Series(model.most_similar(u'余弦函数',topn = 10)))
print('和顶点相近似的词有：')
print(model.most_similar('顶点'))
print('和多边形相近似的词有：')
print(model.most_similar('多边形'))




