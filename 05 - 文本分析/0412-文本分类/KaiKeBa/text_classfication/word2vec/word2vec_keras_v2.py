# _*_ coding:utf8 _*_
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense,Activation
import jieba
import nltk
import numpy as np
from keras import utils
from keras import optimizers
np.random.seed(2018)
BATCH_SIZE = 128
NUM_EPOCHS = 20

# 按照停用词表 特殊字符等过滤词典
def filter_word_set(word_set):
    return word_set

def create_word2inehot(word2index):
    # 构建word2onehot
    word2onehot = {}
    for word in word2index:
        onehot = np.zeros([len(word2index)])
        onehot[word2index[word]] = 1
        word2onehot[word] = onehot
    return word2onehot


def create_word2index(filepath_in):
    data_in = open(filepath_in, 'r',encoding='utf8')
    word_set = set()
    for line in data_in:
        words = jieba.lcut(line.strip())
        for word in words:
            word_set.add(word)
    data_in.close()
    print('word_set is: ',word_set)
    # 过滤单词
    word_set = filter_word_set(word_set)
    # 构建word2index
    word2index = {}
    index = 0
    for word in word_set:
        word2index[word] = index
        index += 1
    return word2index


def create_training_data(filepath_in,word2index):
    data_in = open(filepath_in,'r',encoding='utf8')
    sentences = []
    for line in data_in:
        words = jieba.lcut(line)
        # print('words is: ',words)
        words_index = [word2index[word] for word in words if word in word2index]
        # print('words_index is: ',words_index)
        sentences.append(words_index)
    total = []
    for sentence in sentences:
        print('sentence is: ',sentence)
        triple_list = list(nltk.ngrams(sentence,3))    # 相当于窗口大小 C = 4
        print('list(nltk.trigrams(sentence)) is: ',list(nltk.trigrams(sentence)))
        for triple in triple_list:
            x = []
            y = []
            print('triple is: ',triple)
            # x.append(triple[0])
            # x.append(triple[1])
            # y.append(triple[2])
            # x.append(triple[3])
            # x.append(triple[4])
            x.append(triple[0])
            y.append(triple[1])
            x.append(triple[2])
            total.append((x,y))
            # X.append(x)
            # Y.append(y)
    np.random.shuffle(total)
    X = []
    Y = []
    for trip in total:
        X.append(trip[0])
        Y.append(trip[1])
    print('X is: ',X)
    print('Y is: ',Y)
    return X,Y


def create_model(x_train,y_train,embedding_dim):
    # 模型部分 重要讲解
    model = Sequential()
    model.add(Dense(embedding_dim, input_shape=(x_train.shape[1],)))
    model.add(Activation("relu"))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation("softmax"))
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = optimizers.adam(lr=0.0001)
    model.compile(optimizer='adam', loss="categorical_crossentropy",metrics=["accuracy"])
    # history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
    #                     epochs=NUM_EPOCHS, verbose=1,validation_data=(x_test,y_test))
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS, verbose=1,validation_split=0.3)
    plot(history)


def save_word2index(filepath_out,word2index):
    data_out = open(filepath_out,'w',encoding='utf8')
    for word in word2index:
        data_out.write(word+':'+str(word2index[word])+'\n')
    data_out.close()

def plot(history):
    # plot loss function
    plt.subplot(211)
    plt.title("accuracy")
    plt.plot(history.history["acc"], color="r", label="train")
    plt.plot(history.history["val_acc"], color="b", label="validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="validation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    word2index = create_word2index('./data/人民的名义.txt')
    print('word2index is:',word2index)
    word2onehot = create_word2inehot(word2index)
    # for word in word2index:
    #     print('word is: ',word,word2index[word],word2onehot[word])
    # save_word2index('word2index.txt',word2index)
    # save_word2index('word2onehot.txt',word2onehot)
    X_train,Y_train = create_training_data('./data/人民的名义.txt',word2index)
    print('X_train is: ',X_train)
    X_new = []
    for x in X_train:         # 这里的每个x是一个list
        x_new = np.zeros([len(word2index)])
        for value in x:
            x_new[value] = 1
        X_new.append(x_new)
    x_train = np.array(X_new)
    y_train = np.array(Y_train)
    y_train = utils.to_categorical(y_train)
    print('x_train.shape is: ',x_train.shape)
    print('y_train.shape is: ',y_train.shape)
    embedding_dim = 100
    create_model(x_train,y_train,embedding_dim)





