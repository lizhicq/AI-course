from keras.layers import Embedding,Dense,Conv1D,Input,MaxPooling1D,Flatten,concatenate,Dropout
from keras import Model
def text_cnn(padding_length,vocab_size,embedding_dim,label_num):
    #(batch_size,padding_length)
    seq = Input(shape=(padding_length,))
    #(batch_size,padding_length,embedding_dim)
    emb = Embedding(vocab_size,embedding_dim,input_length=padding_length)(seq)
    filter_sizes = [2,3,4]
    conv_list = []
    for fsz in filter_sizes:
        #(batch_size,padding_length-fsz+1,filters)
        conv1 = Conv1D(filters = 50,kernel_size = fsz )(emb)
        print('conv1.shape is: ',conv1.shape)
        #(batch_size,1,filters)
        pool1 = MaxPooling1D(padding_length-fsz+1)(conv1)
        #(batch_size,filters)
        pool1 = Flatten()(pool1)
        conv_list.append(pool1)
    #(batch_size,3*filters)
    merge = concatenate(conv_list,axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(label_num,activation='sigmoid')(out)
    model = Model(seq,output)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    padding_length = 300
    vocab_size =20000
    embedding_dim = 200
    label_num= 10
    text_cnn(padding_length,vocab_size,embedding_dim,label_num)








