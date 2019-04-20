#encoding=utf-8
import jieba
# sentence = "来到北京清华大学"
sentence = "汪李明来到北京清华大学"
words = jieba.lcut(sentence,cut_all=True)
print(words)
words = jieba.lcut(sentence,cut_all=False)
print(words)