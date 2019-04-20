from keras.layers import Embedding,Dense,Dropout,Input,Conv1D,MaxPooling1D,Flatten,concatenate,Lambda
from keras import Model
import numpy as np
def expand_dim_backend(x):
    from keras import backend
    return backend.expand_dims(x,axis=3)
def text_cnn(padding_length,vocab_size,embedding_dims,filters,label_num):
    #Inputs
    seq = Input(shape=(padding_length,),name='x_seq')
    #Embedding layers
    emb = Embedding(vocab_size,embedding_dims)(seq)
    # conv layers
    convs = []
    filter_sizes = [2,3,4,5]
    for fsz in filter_sizes:
        conv1 = Conv1D(filters,kernel_size=fsz,activation='tanh')(emb)
        pool1 = MaxPooling1D(padding_length-fsz+1)(conv1)
        pool1 = Flatten()(pool1)
        convs.append(pool1)
    merge = concatenate(convs,axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32,activation='relu')(out)
    output = Dense(units=label_num,activation='sigmoid')(output)
    model = Model([seq],output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

# 此时 不能使用一维卷积
def multi_embedding(seq):
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
    padding_length = 300
    vocab_size = 20000
    embedding_dims = 200
    num_filters = 100
    label_num=10
    text_cnn(padding_length,vocab_size,embedding_dims,num_filters,label_num)