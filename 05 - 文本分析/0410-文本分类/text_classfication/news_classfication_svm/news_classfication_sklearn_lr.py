import jieba
import numpy as np
from sklearn import metrics
from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from keras import utils
def get_word_set(filename,min_word_num):
    data_in = open(filename,'r',encoding='utf8')
    word_dict={} # 计算每个单词的频次 后面过滤低频的单词
    label_set = set()
    for line in data_in:
        label,sentence = line.strip().split('\t')
        label_set.add(label) # 添加label
        words = jieba.cut(sentence) # 添加word
        for word in words:
            word = word.strip()
            if word=='': #过滤
                continue
            if word not in word_dict:   # 下面的代码是给每个词计数 比如 吃
                word_dict[word] = 1
            else:
                num = word_dict[word]
                num+=1
                word_dict[word] = num
                # word_set.add(word)
    data_in.close()

    print('len(word_dict) is: ',len(word_dict))
    # 过滤掉word_dict中低频的词
    # min_num = 30
    word_dict_new,word_set = filter_word_dict(word_dict,min_word_num)
    print('len(word_dict_new) is: ',len(word_dict_new))
    return word_set

def filter_word_dict(word_dict,min_num):
    word_dict_new = {}
    word_set = set()
    for word,num in word_dict.items():
        if num>=min_num:
            word_dict_new[word] = num
            word_set.add(word)
    return word_dict_new,word_set


def generate_training_data(filename_in,word_set):
    data_in = open(filename_in,'r',encoding='utf8')
    sentence_new_list = []
    label_list = []
    for line in data_in:
        label,sentence = line.strip().split('\t')
        label_list.append(label)
        words = jieba.cut(sentence)
        words_new = [] # 用来对words进行过滤的 保证只有一定词频以上的单词才能作为特征
        for word in words:
            if word in word_set:
                words_new.append(word)
        # print('len(words_new) is: ',len(words_new))
        sentence_new = ' '.join(v for v in words_new)
        sentence_new_list.append(sentence_new)
    print('sentence_new_list is: ',sentence_new_list)
    print('len(sentence_new_list) is: ',len(sentence_new_list))
    vectorizer = CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tf_matrix= vectorizer.fit_transform(sentence_new_list)
    print('tf matrix is: ',tf_matrix)
    tfidf = transformer.fit_transform(tf_matrix)#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    print('tfidf matrix is: ',tfidf)
    word = vectorizer.get_feature_names()#获取词袋模型中的所有词语
    print('word is: ',word)
    print('word_set is: ',word_set)
    weight = tfidf.toarray() # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print('weight.shape is: ',weight.shape)
    total = []
    for i in range(len(weight)):    # for遍历所有文本，
        x = []
        y = []
        for j in range(len(weight[0])): # 遍历每一类文本下的词语权重
            x.append(weight[i][j])
        if i<=500:
            y.append(0)
        else:
            y.append(1)
        total.append((x,y))
    np.random.shuffle(total)

    training_feas = []
    training_labels = []
    for tup in total:
        training_feas.append(tup[0])
        training_labels.append(tup[1])
    training_feas = np.array(training_feas)
    training_labels = np.array(training_labels)

    return training_feas,training_labels

def split_training_data(training_feas,training_labels):
    trainSize = int(2*training_feas.shape[0]/3)
    # print('trainSize is {0}'.format(trainSize))
    # print('testSize is {0}'.format(X.shape[0]-trainSize))
    x_train = training_feas[0:trainSize,]
    y_train = training_labels[0:trainSize,]
    x_test = training_feas[trainSize:,]
    y_test = training_labels[trainSize:,]
    return x_train,y_train,x_test,y_test

if __name__ == '__main__':
    training_file = 'data/cnews_train1.txt'
    min_word_num = 30
    word_set = get_word_set(training_file,min_word_num) # 获取原始文件中词语出现次数大于min_word_num的词语集合
    print('len(word_set) is: ',len(word_set))
    # 这是使用tf-idf的核心部分
    training_feas,training_labels = generate_training_data(training_file,word_set)
    print('training_feas is: ',training_feas)
    print('training_labels is: ',training_labels)

    print('training_labels.shape is: ',training_labels.shape)
    print('training_feas.shape is: ',training_feas.shape)
    # 按照一定比例切分训练集测试集
    x_train,y_train,x_test,y_test = split_training_data(training_feas,training_labels)

    model = linear_model.LogisticRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred_porb = model.predict_proba(x_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    print('acc is: ',acc)
    data_out = open('res_lr.txt','w',encoding='utf8')
    for i in range(0,len(y_test)):
        # print(y_test[i][0],y_pred[i])
        data_out.write(str(y_test[i][0])+'\t'+str(y_pred[i])+'\t'+str(y_pred_porb[i])+'\n')
    data_out.close()
