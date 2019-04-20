# 训练新闻数据集的词向量
import gensim
import pandas as pd
import jieba
subject = "gzsx"
dim = 100

def read_data():
    print("正在准备数据...")
    data_in =  open('data/cnews.txt', 'r',encoding='utf8')
    sentences_list = []
    for line in data_in:
        label,sentences = line.strip().split('\t')
        words = jieba.lcut(sentences)
        sentences_list.append(words)
    return sentences_list


sentences_list = read_data()
print('sentences_list is: ',sentences_list)
# sentences = [['first', 'sentence'], ['second', 'sentence','is']]
print("正在进行模型训练...")
model = gensim.models.Word2Vec(sentences_list, min_count=1, size=dim)
print("模型训练完毕...")
savepath = "news_word2vec_{1}.model".format(subject, dim)
model.save(savepath)
print("保存模型完毕！")
# #
# #
# savepath = "word2vec_200.model"
#

# #测试
# print('-----------------------------------------')
# model = gensim.models.Word2Vec.load(savepath)
# print('和篮球相近似的词有：')
# print(pd.Series(model.most_similar(u'篮球',topn = 10)))
# print('和参数方程相近似的词有：')
# print(pd.Series(model.most_similar(u'参数方程',topn = 10)))
# print('和余弦函数相近似的词有：')
# print(pd.Series(model.most_similar(u'余弦函数',topn = 10)))
# print('和顶点相近似的词有：')
# print(model.most_similar('顶点'))
# print('和多边形相近似的词有：')
# print(model.most_similar('多边形'))




