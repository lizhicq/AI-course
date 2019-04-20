from keras.layers import Embedding,Dense,GlobalAveragePooling1D,Input,Activation
from keras.models import Model,Sequential
def text_fastext(vocab_size,embedding_dim,label_num):
    model = Sequential()
    model.add(Embedding(vocab_size,embedding_dim,input_length=300))
    model.add(GlobalAveragePooling1D())
    # model.add(Dense(label_num*5))  # 如果加上这一层，效果不好
    model.add(Dense(label_num))
    model.add(Activation(activation='sigmoid'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

# def text_fastext(vocab_size,embedding_dim,label_num):
#     seq = Input(shape=[300])     # 指定shape
#     print('seq.shape is: ',seq.shape)
#
#     emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(seq)
#     print('emb.shape is: ',emb.shape)
#
#     output = Dense(units=10,activation='relu')(emb)
#     print('output1.shape is: ',output.shape)
#     # model = Sequential()
#     # model.add(Embedding(vocab_size,embedding_dim))
#     # model.add(GlobalAveragePooling1D())
#     # model.add(Dense(label_num*5))
#     # model.add(Dense(label_num))
#     # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#     # model.summary()
#     return model

if __name__ == '__main__':
    text_fastext(30000,200,3)
