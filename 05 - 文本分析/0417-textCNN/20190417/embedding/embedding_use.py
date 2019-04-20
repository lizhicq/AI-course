import numpy as np
import gensim
from keras import Sequential,Model
from keras.layers import Input,Embedding,LSTM,Bidirectional,Dense
from keras import optimizers

def create_embedding(word2index, num_words, word2vec_model):
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, index in word2index.items():
        embedding_vector = word2vec_model[word]
        embedding_matrix[index] = embedding_vector
    return embedding_matrix



EMBEDDING_DIM = 200 # 词向量的维度
word_index = {'直线':0,'参数方程':1}
num_words = 2
word2vec_model_path = 'word2vec_200.model'
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)
embedding_matrix = create_embedding(word_index, num_words, word2vec_model)
print('embedding_matrix is: ',embedding_matrix)
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            # embeddings_initializer=Constant(embedding_matrix),
                            # input_length=MAX_SEQUENCE_LENGTH,
                            weights= [embedding_matrix],
                            trainable=False)

model = Sequential()
model.add(embedding_layer)
input_array = np.random.randint(low=0, high=2, size=(32, 2))
print('input_array is: ',input_array)
model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
model.summary()
print('output_array is: ',output_array)