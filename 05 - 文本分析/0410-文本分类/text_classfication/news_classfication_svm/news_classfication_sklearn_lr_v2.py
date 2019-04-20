import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def get_word_set(filepath_in,min_word_count):
    data_in = open(filepath_in,'r',encoding='utf8')
    label_set = set()
    word_num_dict = {}
    for line in data_in:
        label,sentence = line.strip().split('\t')
        label_set.add(label)  #添加标签
        words = jieba.lcut(sentence)
        for word in words:
            word = word.strip()
            if word not in word_num_dict:
                word_num_dict[word] = 1
            else:
                num = word_num_dict[word]
                num+=1
                word_num_dict[word] = num
    # print(label_set)
    # print(word_num_dict)
    data_in.close()

    word_num_dict_new,word_set = filter(word_num_dict,min_word_count)
    return word_set






def filter(word_num_dict,min_word_count):
    word_num_dict_new = {}
    word_set = set()
    for word,num in word_num_dict.items():
        if num>= min_word_count:
            word_num_dict_new[word] = num
            word_set.add(word)
    return word_num_dict_new,word_set


def generate_training_file(training_file,word_set):
    data_in = open(training_file,'r',encoding='utf8')
    label_list = []
    sentence_new_list = []
    for line in data_in:
        label,sentence = line.strip().split('\t')
        label_list.append(label)
        words = jieba.lcut(sentence)
        words_new = []
        for word in words:
            if word in word_set:
                words_new.append(word)
        sentence_new = ' '.join(v for v in words_new)
        sentence_new_list.append(sentence_new)
        # print('sentence_new_list is: ',sentence_new_list)
    countvectorizer = CountVectorizer()
    countvectorizer.fit(sentence_new_list)
    tf_matrix = countvectorizer.transform(sentence_new_list)

    tfidftransformer = TfidfTransformer()
    tfidftransformer.fit(tf_matrix)
    tfidf_matrix = tfidftransformer.transform(tf_matrix)
    print('tfidf_matrix is: ',tfidf_matrix.toarray())
    weight = tfidf_matrix.toarray()
    total = []
    for i in range(0,len(weight)):
        x = []
        y = []
        for j in range(len(weight[0])):
            x.append(weight[i][j])
        if i<=500:
            y.append(0)
        else:
            y.append(1)
        total.append((x,y))
    np.random.shuffle(total)
    X = []
    Y = []
    for trip in total:
        X.append(trip[0])
        Y.append(trip[1])
    return np.array(X),np.array(Y)






def split_training_data(X,Y):
    training_size = int(2*X.shape[0]/3)
    x_train = X[0:training_size,]
    y_train = Y[0:training_size,]
    x_test = X[training_size:,]
    y_test = Y[training_size:,]
    return x_train,y_train,x_test,y_test




if __name__ == '__main__':
    training_file = 'data/cnews_train1.txt'
    min_word_count = 30
    word_set = get_word_set(training_file,min_word_count)
    # print(word_set)
    X,Y = generate_training_file(training_file,word_set)
    print(X.shape)
    print(Y.shape)
    x_train,y_train,x_test,y_test = split_training_data(X,Y)
    # model = LogisticRegression()
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    print('acc is: ',acc)
    for i in range(0,len(y_test)):
        print(y_test[i][0],y_pred[i])






