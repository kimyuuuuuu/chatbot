import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Komoran
import matplotlib.pyplot as plt
from chatbot2_sim.utils.Preprocess import Preprocess

d = pd.read_csv("chatbot2_sim/model/intent/total_train_data_4.csv")

print("data shape: ", d.shape)

text_list = d['query'].tolist()
token = []

p = Preprocess()

for i in text_list :
  dt = p.pos(i)
  token.append(p.get_keywords(dt))
#print(token[:10])
num_tokens = [len(tokens) for tokens in token]
num_tokens = np.array(num_tokens)

print("토큰 길이 평균:", np.mean(num_tokens))
print("토큰 길이 최대:", np.max(num_tokens))
print("토큰 길이 표준편차:", np.std(num_tokens))

# plt.title('text length')
# plt.hist(num_tokens, bins=100)
# plt.xlabel('length of data')
# plt.ylabel('number of data')
# plt.show()

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) >= max_len):
            cnt = cnt + 1
        
    #print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))
    print("전체 동아리 중 %d개 이상인 샘플 개수: "%(max_len), cnt)

below_threshold_len(25, token)
below_threshold_len(15, token)
below_threshold_len(10, token)
below_threshold_len(7, token)
