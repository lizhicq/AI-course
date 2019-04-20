from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# 下面的代码是计算词频矩阵
corpus = ['I come to China to travel','I like to travel in china','I like tea ']
stop_words = set()
# stop_words.add('in')
countvectorizer = CountVectorizer(stop_words=stop_words)
countvectorizer.fit(corpus)
count = countvectorizer.transform(corpus)
print('document-term matrix is:')
print(count)
print(count.toarray())
print('词典如下：')
print(countvectorizer.get_feature_names())
print('获取停用词表如下：')
print(countvectorizer.get_stop_words())

# 下面的代码是计算tf-idf矩阵
transformer = TfidfTransformer()
tfidf_matrix = transformer.fit_transform(count)
print(tfidf_matrix.toarray())