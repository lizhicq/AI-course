import numpy as np
import jieba
from keras.preprocessing import sequence
from keras import utils
def split_dataset(filename_in):
    data_in = open(filename_in,'r',encoding='utf8')
    label_sentences_dict = {}  # 定义一个标签：样本的字典
    for line in data_in:
        label,sentence = line.strip().split('\t')
        if label not in label_sentences_dict:
            lst = []
            lst.append(sentence)
            label_sentences_dict[label] = lst
        else:
            lst = label_sentences_dict[label]
            lst.append(sentence)
            label_sentences_dict[label] = lst
    # print(len(label_sentences_dict))
    data_in.close()
    return label_sentences_dict

def save2file(label_sentences_dict,train_ratio,training_file,testing_file):
    data_out_training = open(training_file,'w',encoding='utf8')
    data_out_testing = open(testing_file,'w',encoding='utf8')
    for label in label_sentences_dict:
        sentences = label_sentences_dict[label]
        np.random.shuffle(sentences)
        for i in range(0,len(sentences)):
            if i <int(train_ratio*len(sentences)):
                data_out_training.write(label+':'+sentences[i]+'\n')
            else:
                data_out_testing.write(label+':'+sentences[i]+'\n')
    data_out_training.close()
    data_out_testing.close()

# 返回字典集合 标签集合 过滤低频词
def create_word2index(train_file,min_num):
    data_in = open(train_file,'r',encoding='utf8')
    label_set = set()
    # word_set = set()
    word_dict={}
    for line in data_in:
        label,sentence = line.strip().split('\t')
        label_set.add(label) # 添加label
        words = jieba.cut(sentence) # 添加word
        for word in words:
            word = word.strip()
            if word == '' or word == '\t' or word == ',' or word == ':': # 过来特殊符号 可自定义
                continue
            if word not in word_dict:
                word_dict[word] = 1
            else:
                num = word_dict[word]
                num+=1
                word_dict[word] = num
            # word_set.add(word)

    data_in.close()

    print('word_dict is: ',word_dict)
    # 过滤掉word_dict中低频的词
    word_dict_new,word_set = filter_word_dict(word_dict,min_num)
    print('word_dict_new is: ',word_dict_new)
    word2index = {}
    index = 0
    for word in word_set:
        word2index[word] = index
        index+=1

    label2index = {}
    index = 0
    for label in label_set:
        label2index[label] = index
        index+=1
    if '' in word_set:
        print('exist')
    else:
        print('no exist')
    return word2index,label2index


def filter_word_dict(word_dict,min_num):
    word_dict_new = {}
    word_set = set()
    for word,num in word_dict.items():
        if num>=min_num:
            word_dict_new[word] = num
            word_set.add(word)
    return word_dict_new,word_set


def create_trainingdata(training_file,word2index,label2index,padding_length = 300):
    data_in = open(training_file,'r',encoding='utf8')
    training_data = []
    for line in data_in:
        label,sentence = line.strip().split('\t')
        words = jieba.cut(sentence)
        feas = [word2index[word]for word in words if word in word2index]
        label = [label2index[label]]
        training_data.append((feas,label)) # 将特征和标签放一起读取 然后随机打乱
    data_in.close()

    np.random.shuffle(training_data)

    training_feas = []
    training_labels = []
    for feas, label in training_data:
        training_feas.append(feas)
        training_labels.append(label)
    # print('training_feas is: ',training_feas)
    # print('training_labels is: ',training_labels)

    training_feas = sequence.pad_sequences(training_feas,maxlen=padding_length,padding='post', truncating='post')
    training_labels = utils.to_categorical(training_labels,num_classes=10)
    # print('after, training_feas is: ',training_feas)
    # print('after, training_labels is: ',training_labels)
    training_feas = np.array(training_feas)
    training_labels = np.array(training_labels)
    return training_feas,training_labels






if __name__ == '__main__':
    # label =
    label_sentences_dict = split_dataset('cnews_test.txt')
    train_ratio = 0.1
    save2file(label_sentences_dict,train_ratio,'cnews_test1.txt','cnews_test2.txt')

    # for label in label_sentences_dict:
    #     print(label)
    #     print(len(label_sentences_dict[label]))