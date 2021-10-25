import jieba
from gensim.models import word2vec

with open('test.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('testsplit.txt', 'a', encoding='utf-8') as ff:
           ff.write(' '.join(seg_list)) # 词汇用空格分开


# 加载语料
sentences = word2vec.Text8Corpus('testsplit.txt')

# 训练模型
model = word2vec.Word2Vec(sentences)

model.save('model.model')

# 加载模型
model = word2vec.Word2Vec.load('model.model')

print(model.wv['词向量'])