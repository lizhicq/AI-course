#encoding=utf-8
import jieba
# sentence = "我来到北京清华大学"
sentence = "故宫的著名景点包括乾清宫、太和殿和紫禁城，我最喜欢乾清宫，我经常和王李明一起去玩"
jieba.load_userdict('word_dict.txt')
words = jieba.lcut(sentence)
print(words)