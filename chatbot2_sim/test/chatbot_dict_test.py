import pickle
from chatbot2_sim.utils.Preprocess import Preprocess

# 단어사전 불러오기
f = open("chatbot2_sim/train_tools/dict/chatbot_dict.bin", "rb")
word_index = pickle.load(f)
f.close()

sent = "루시우스에 대해 알려줘 ㅋㅋ"

# 전처리 객체 생성
p = Preprocess(userdic="chatbot2_sim/utils/user_dic.tsv")

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태그 없이 키워드 출력
keywords = p.get_keywords(pos, without_tag=True)
for word in keywords : 
  try :
    print(word, word_index[word])
  except KeyError :
    # 해당 단어가 사전에 없는 경우 OOV 처리
    print(word, word_index['OOV'])
