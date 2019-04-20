# class TextCNN():
from keras.models import Sequential
import keras
from keras.models import Model
from keras.layers import Embedding,Conv1D,Input,MaxPooling1D,Flatten,Dense,Dropout,Add,Conv2D,Lambda
from keras.layers import concatenate
import numpy as np
import tensorflow as tf
from keras import backend
from keras import metrics

def expand_dim_backend(x):
    from keras import backend
    return backend.expand_dims(x,axis=3)

def squeeze_backend(conv2):
    from keras import backend
    return backend.squeeze(conv2,axis=2)


def text_cnn(padding_length,vocab_size,embedding_dim,num_filters,label_num):
    # 在rnn中，seq_length是可变的，但是在textcnn中 是不可变的，需要构成一个固定大小的矩阵
    #Inputs
    # seq = Input(shape=[maxlen],name='x_seq')
    seq = Input(shape=(padding_length,))
    print('seq.shape is: ',seq.shape)
    #Embedding layers
    # emb = Embedding(vocab_size,embedding_dim)(seq)
    # print('emb.shape is: ',emb.shape)
    # emb = Lambda(expand_dim_backend)(emb)
    # multi Embedding layers
    emb = multi_embedding(seq,vocab_size,embedding_dim)

    print('after, emb.shape is: ',emb.shape)
    # conv layers
    convs = []
    filter_sizes = [2,3,4,5]
    for fsz in filter_sizes:
        conv2 = Conv2D(filters=num_filters,kernel_size=(fsz,embedding_dim))(emb)
        print('conv2.shape is: ',conv2.shape)
        # conv2 = keras.backend.squeeze(conv2,axis=2)
        conv2 = Lambda(squeeze_backend)(conv2)
        print('after, conv2.shape is: ',conv2.shape)
        # conv2 = tf.transpose(conv2,perm=[0,2,1])
        # print('conv2.shape is: ',conv2.shape)
        pool1 = MaxPooling1D(pool_size=padding_length-fsz+1)(conv2)
        print('pool1.shape is: ',pool1.shape)
        pool1 = Flatten()(pool1)  # 完全压扁 除了batch_size
        convs.append(pool1)
    merge = concatenate(convs,axis=1)  #  输入是一个tensor的list
    print('merge.shape is: ',merge.shape)
    out = Dropout(0.5)(merge)
    print('out.shape is: ',out.shape)
    output = Dense(32,activation='relu')(out)

    # output = Dense(units=label_num,activation='sigmoid')(output)
    output = Dense(units=label_num,activation='softmax')(output)
    print('output.shape is: ',output.shape)
    print( 'seq.shape is: ',seq.shape)
    # model = Model([seq],output)
    model = Model(seq,output)
    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['categorical_accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    # model.predict()
    return model

def multi_embedding(seq,vocab_size,embedding_dims):
    emb1 = Embedding(vocab_size,embedding_dims)(seq)
    emb2 = Embedding(vocab_size,embedding_dims)(seq)
    print('emb1.shape is: ',emb1.shape)
    emb1 = Lambda(expand_dim_backend)(emb1)
    emb2 = Lambda(expand_dim_backend)(emb2)
    emb_list = []
    emb_list.append(emb1)
    emb_list.append(emb2)
    emb_new = concatenate(emb_list,axis=3)
    print('emb_new is: ',emb_new)
    return emb_new

if __name__ == '__main__':
    text_cnn(100,100,100,100,100)