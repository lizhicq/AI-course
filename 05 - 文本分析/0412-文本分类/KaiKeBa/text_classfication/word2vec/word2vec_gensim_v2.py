import gensim  # word2vec doc2vec word2index vocabulary

def read_data_file(filepath_in):
    data_in = open(filepath_in,'r',encoding='utf8')
    feas_list = []
    count = 0
    for line in data_in:
        count += 1
        if count>=100:
            break
        stid,sentence = line.strip().split('\t')
        feas = sentence.split(':')
        feas_list.append(feas)
    data_in.close()
    return feas_list


if __name__ == '__main__':
    # filepath_in = 'set_0204.txt'
    # feas_list = read_data_file(filepath_in) # [[],[],[]]
    # # print(feas_list)
    # embedding_dim = 50
    # model = gensim.models.Word2Vec(feas_list,embedding_dim)
    # model.save('test_word2vec.model')

    model = gensim.models.Word2Vec.load('test_word2vec.model')
    print(model.most_similar('二次根式',topn=10))





