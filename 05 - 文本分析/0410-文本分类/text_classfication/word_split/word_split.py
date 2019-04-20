#encoding=utf-8
import jieba
sentence = "我来到北京清华大学"#'他创造了这个短语'
words = jieba.cut(sentence)  # 生成生成器
print(words)
# for word in words:
#     print(word)

words = jieba.lcut(sentence)    # 生成list
print(words)
# for word in words:
#     print(word)


# 全模式
text = "我来到北京清华大学"
seg_list = jieba.cut(text, cut_all=True)
print(u"[全模式]: ", "/ ".join(seg_list))

#精确模式
seg_list = jieba.cut(text, cut_all=False)
print(u"[精确模式]: ", "/ ".join(seg_list))

#默认是精确模式
seg_list = jieba.cut(text)
print(u"[默认模式]: ", "/ ".join(seg_list))


# 添加自定义词典
jieba.load_userdict("word_dict.txt")