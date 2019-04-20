from text_classfication.news_classfication_fasttext import preprocess_data
from text_classfication.news_classfication_fasttext import fasttext
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


    # vocab_size = len(word2index)
    # embedding_dim = 200
    # model = fasttext.text_fastext(vocab_size, embedding_dim, len(label2index))
    # # # 通过fit的callbacks参数将回调函数传入模型中，这个参数接收一个回调函数列表，你可以传入任意个回调函数
    # # callback_lists = [
    # #     keras.callbacks.EarlyStopping(monitor = 'acc',  # 监控模型的验证精度
    # #                                   patience = 1,),   # 如果精度在多于一轮的时间（即两轮）内不再改善，就中断训练
    # #     # ModelCheckpoint用于在每轮过后保存当前权重
    # #     keras.callbacks.ModelCheckpoint(filepath = 'news_classfication_textcnn.h5', # 目标文件的保存路径
    # #                                     # 这两个参数的意思是，如果val_loss没有改善，那么不需要覆盖模型文件，
    # #                                     # 这就可以始终保存在训练过程中见到的最佳模型
    # #                                     monitor = 'val_loss', save_best_only = True,)
    # # ]
    # # history = model.fit(training_feas,training_labels,batch_size=256,epochs=30,validation_split=0.3,callbacks=callback_lists)
    # history = model.fit(training_feas,training_labels,batch_size=256,epochs=30,validation_split=0.3)
    # model.save('news_classfication_textcnn.h5')
    # plot(history)

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
