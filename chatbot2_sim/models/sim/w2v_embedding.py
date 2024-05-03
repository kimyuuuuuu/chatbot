from gensim.models import Word2Vec
from konlpy.tag import Komoran
import pandas as pd
import time

def read_data(filename):
  data = pd.read_excel(filename)
  # 'query' 열에서 데이터 추출
  queries = data['질문(Query)'].tolist()
  return queries

start = time.time()

# 데이터 읽어오기
print('1) 질문 데이터 읽어오기')
query_data = read_data('./chatbot2_sim/models/sim/train_data_5.xlsx')
print(len(query_data))
print('1) 데이터 읽기 완료:', time.time()-start)

# 문장 단위로 명사 추출
print('2) 명사 추출')
komoran = Komoran()
docs = [komoran.nouns(sentence[1]) for sentence in query_data]
print('2) 명사 추출 완료: ', time.time() - start)

# W2V 모델 학습
print('3) Word2Vector 모델 학습')
model = Word2Vec(sentences=docs, vector_size=200, window=4, hs=1, min_count=2, sg=1)
print('3) Word2Vec 모델 학습 완료', time.time() - start)

# 모델 저장
print('4) 모델 저장')
model.save('./chatbot2_sim/models/sim/W2V.model')
print('4) 모델 저장 완료', time.time() - start)

# 모델의 단어 벡터만 저장
model.wv.save("./chatbot2_sim/models/sim/word_vectors.kv")

# 학습된 말뭉치 수, 코퍼스 내 전체 단어 수
print("corpus_count: ", model.corpus_count)
print("corpus_total_words : ", model.corpus_total_words)