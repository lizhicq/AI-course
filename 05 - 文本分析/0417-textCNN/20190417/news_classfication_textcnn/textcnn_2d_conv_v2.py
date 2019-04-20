from keras.layers import Embedding,Dense,Conv2D,Input,MaxPooling1D,Flatten,concatenate,Dropout,Lambda
from keras import Model
def expand_dim_backend(x):
    from keras import backend
    return backend.expand_dims(x,axis=3)

def squeeze_backend(x):
    from keras import backend
    return backend.squeeze(x,axis=2)

def multi_embedding(vocab_size,embedding_dim,seq):
    #(batch_size,padding_length,embedding_dim)
    emb1 = Embedding(vocab_size,embedding_dim,input_length=padding_length)(seq)
    emb2 = Embedding(vocab_size,embedding_dim,input_length=padding_length)(seq)
    #(batch_size,padding_length,embedding_dim,1)
    emb1 = Lambda(expand_dim_backend)(emb1)
    emb2 = Lambda(expand_dim_backend)(emb2)
    print('emb2.shape is: ',emb2.shape)
    emb_list = []
    emb_list.append(emb1)
    emb_list.append(emb2)
    emb = concatenate(emb_list,axis=3)
    return emb

    #(batch_size,padding_length,embedding_dim,2)



def text_cnn(padding_length,vocab_size,embedding_dim,label_num):
    #(batch_size,padding_length)
    seq = Input(shape=(padding_length,))
    # #(batch_size,padding_length,embedding_dim)
    # emb = Embedding(vocab_size,embedding_dim,input_length=padding_length)(seq)
    # #(batch_size,padding_length,embedding_dim,1)
    # emb = Lambda(expand_dim_backend)(emb)

    #(batch_size,padding_length,embedding_dim,2)
    emb = multi_embedding(vocab_size,embedding_dim,seq)

    filter_sizes = [2,3,4]
    conv_list = []
    for fsz in filter_sizes:
        #(batch_size,padding_length-fsz+1,1,filters)
        conv1 = Conv2D(filters = 50,kernel_size = (fsz,embedding_dim) )(emb)
        #(batch_size,padding_length-fsz+1,filters)
        conv1 = Lambda(squeeze_backend)(conv1)
        #(batch_size,1,filters)
        pool1 = MaxPooling1D(padding_length-fsz+1)(conv1)
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