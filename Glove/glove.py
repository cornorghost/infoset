from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = 'glove.840B.300d.txt'
word2vec_output_file = 'glove.840B.300d.word2vec.txt'
(count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
print(count, '\n', dimensions)

# 加载模型
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# 如果希望直接获取某个单词的向量表示，直接以下标方式访问即可
word_vec = glove_model['cat']
print(word_vec)