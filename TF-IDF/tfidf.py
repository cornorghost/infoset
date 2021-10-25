# coding=utf-8
mydoclist = [u'温馨 提示 ： 家庭 畅享 套餐 介绍 、 主卡 添加 / 取消 副 卡 短信 办理 方式 , 可 点击 文档 左上方  短信  图标 即可 将 短信 指令 发送给 客户',
u'客户 申请 i 我家 ， 家庭 畅享 计划  后 ， 可 选择 设置 1 - 6 个 同一 归属 地 的 中国移动 网 内 号码 作为 亲情 号码 ， 组建 一个 家庭 亲情 网  家庭 内 ',
u'所有 成员 可 享受 本地 互打 免费 优惠 ， 家庭 主卡 号码 还 可 享受 省内 / 国内 漫游 接听 免费 的 优惠']

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df = 1)
tfidf_matrix = tfidf_vectorizer.fit_transform(mydoclist)
str=''
for i in tfidf_vectorizer.vocabulary_:
    str+=' '+i
print(str)
print(tfidf_matrix.todense())

# new_docs = [u'一个 客户 号码 只能 办理 一种 家庭 畅享 计划 套餐 ， 且 只能 加入 一个 家庭网']
# new_term_freq_matrix = tfidf_vectorizer.transform(new_docs)
# print(tfidf_vectorizer.vocabulary_,type(tfidf_vectorizer.vocabulary_))
# str=''
# for i,j in sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda d: d[1]):
#     str+=' '+i
# print(str)
# print([ v for v in sorted(tfidf_vectorizer.vocabulary_.values())])
# print (sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda d: d[1]))
 
 
# print (new_term_freq_matrix.todense())