from text_classfication.news_classfication_fasttext  import preprocess_data
from keras.models import load_model

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

if __name__ == '__main__':
    predict('cnews_test.txt')  #其他的测试集 太大 运行会卡
    # 结果：[0.471332097530365, 0.896222222328186]
