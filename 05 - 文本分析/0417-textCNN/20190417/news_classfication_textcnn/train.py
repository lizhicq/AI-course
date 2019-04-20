from text_classfication.news_classfication_textcnn import preprocess_data
from text_classfication.news_classfication_textcnn import textcnn_2d_conv,textcnn_1d_conv,textcnn_2d_conv_multi_emdedding
import matplotlib.pyplot as plt
import keras
def save2file(word2index,filename_out):
    data_out = open(filename_out,'w',encoding='utf8')
    for word,index in word2index.items():
        data_out.write(word+':'+str(index)+'\n')
    data_out.close()

def train(training_file):
    min_num = 10    # 单词最少出现的次数
    word2index,label2index = preprocess_data.create_word2index(training_file,min_num)
    print('word2index is: ',word2index)
    print('label2index is: ',label2index)
    # 此处需要把word2index和label2index存成文本
    save2file(word2index,'word2index.txt')
    save2file(label2index,'label2index.txt')

    # 读取训练数据集 返回可以直接入模型的特征和标签
    training_feas,training_labels = preprocess_data.create_trainingdata(training_file,word2index,label2index)
    print('training_feas.shape is: ',training_feas.shape)
    print('training_labels.shape is: ',training_labels.shape)


    vocab_size = len(word2index)
    embedding_dim = 200
    num_filters = 50  # 卷积核的个数
    padding_length = 300
    model = textcnn_2d_conv.text_cnn(padding_length, vocab_size, embedding_dim, num_filters, len(label2index))
    model = textcnn_1d_conv.text_cnn(padding_length, vocab_size, embedding_dim, num_filters, len(label2index))
    model = textcnn_2d_conv_multi_emdedding.text_cnn(padding_length, vocab_size, embedding_dim, num_filters, len(label2index))
    history = model.fit(training_feas,training_labels,batch_size=256,epochs=10,validation_split=0.3)
    model.save('news_classfication_textcnn.h5')
    plot(history)

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
    train('cnews_train.txt')  # 5000条训练数据
