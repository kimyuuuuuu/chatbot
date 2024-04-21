#
# 챗봇에서 사용하는 사전 파일 생성
#
from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data


# 말뭉치 데이터 가져오기
corpus_data = read_corpus_data('C:\\chatbot\\train_tools\\dict\\corpus_new.txt')
# corpus_data = corpus_data[:3]
# print(corpus_data)
# print(corpus_data[1])

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess()
dict = []
for c in corpus_data:
    try :
        pos = p.pos(c[1]) # pos : 형태소 분석 , 토큰과 품사 태그 쌍 return 
        for k in pos:
            dict.append(k[0])
    except Exception as e :
        print(e)
        print(c) # index out of range 탐색용 
                 # data에 오류가 있었음. 수정 완료

# 사전에 사용될 word2index 생성
# 사전의 첫번 째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

# 사전 파일 생성
f = open("./chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()