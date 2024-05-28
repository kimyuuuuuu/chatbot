import matplotlib.pyplot as plt
plt.interactive(False)
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from chatbot1_ner.utils.Preprocess import Preprocess
# 학습 파일 불러오기
def read_file(file_name):
    sents = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if l[0] == ';' and lines[idx + 1][0] == '$':
                this_sent = []
            elif l[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif l[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(l.split()))
    return sents

# 학습용 말뭉치 데이터를 불러옴
#corpus = read_file('./chatbot1_ner/models/ner/테스트용 품사없는 전체 NER2.txt')
corpus = read_file('./chatbot1_ner/models/ner/NER_add.txt')

# 말뭉치 데이터에서 단어와 BIO 태그만 불러와 학습용 데이터셋 생성
sentences, tags = [], []
for t in corpus[2185:2187]:
    tagged_sentence = []
    sentence, bio_tag = [], []
    for w in t:
        print(w[1], w[2])
        tagged_sentence.append((w[1], w[2]))
        sentence.append(w[1])
        bio_tag.append(w[2])
    
    sentences.append(sentence)
    tags.append(bio_tag)

