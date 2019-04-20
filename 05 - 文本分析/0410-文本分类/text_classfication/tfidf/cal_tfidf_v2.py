from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
corpus = ['I come to China to travel','I like to travel in China','I like tea ']
stop_words = set()
stop_words.add('China')
countvecorizer = CountVectorizer(stop_words=stop_words)
countvecorizer.fit(corpus)
count =countvecorizer.transform(corpus)
print(count)
print(count.toarray())  # m*n
print(countvecorizer.get_feature_names())

# tf-idf

tfidfTransformer = TfidfTransformer()
tfidfTransformer.fit(count)
tfidf_matrix = tfidfTransformer.transform(count)
print(tfidf_matrix.toarray())