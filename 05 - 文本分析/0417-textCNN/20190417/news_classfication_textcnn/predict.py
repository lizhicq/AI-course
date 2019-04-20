from text_classfication.news_classfication_bilstm  import preprocess_data
from keras.models import load_model
import numpy as np
np.set_printoptions(threshold=1e6)
def load_word2index(filename_in):
    data_in = open(filename_in,'r',encoding='utf8')
    word2index = {}
    for line in data_in:
        print(line)
        word,index = line.strip().split(':')
        word2index[word] = index
    return word2index


def predict(predict_file):
    word2index = load_word2index('word2index.txt')
    label2index = load_word2index('label2index.txt')
    testing_feas,testing_labels = preprocess_data.create_trainingdata(predict_file,word2index,label2index)

    model = load_model('news_classfication_textcnn.h5')
    score = model.evaluate(testing_feas,testing_labels)   # Returns the loss value & metrics values
    print(score)

    # # 打印概率输出结果
    # preds = model.predict(testing_feas)
    # # print('preds is: ',preds)
    # print('np.argmax(preds,axis=1) is: ',np.argmax(preds,axis=1))
    # print('np.argmax(preds,axis=1) is: ',np.argmax(testing_labels,axis=1))
    # y_pred = np.argmax(preds,axis=1)
    # y_true = np.argmax(testing_labels,axis=1)
    # for i in range(0,len(y_pred)):
    #     print(y_pred[i],y_true[i])

if __name__ == '__main__':
    predict('cnews_test.txt')  #其他的测试集 太大 运行会卡
    # [0.2821743139690823, 0.9155555554495918] [0.28857217131720647, 0.9153333332273695] textcnn_conv2d
    # [0.2487703538073434, 0.9253333334392971] textcnn_conv2d_multi_embedding

