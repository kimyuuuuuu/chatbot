from chatbot2_sim.utils.Preprocess import Preprocess
from chatbot2_sim.models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='./chatbot2_sim/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot2_sim/utils/user_dic.tsv')

intent = IntentModel(model_name='./chatbot2_sim/models/intent/intent_model.h5', preprocess=p)
query = ["원주대 중앙동아리에는 뭐가 있어?", "운동 동아리 알려줘", "CCC 회비 얼마야", "프레이즈에 대해 알려줘", "취업동아리 어떻게 만들어"]

for i in query :
  predict = intent.predict_class(i)
  predict_label = intent.labels[predict]

  print(i)
  print("의도 예측 클래스 : ", predict)
  print("의도 예측 레이블 : ", predict_label)
