from chatbot2_sim.utils.Preprocess import Preprocess
from chatbot2_sim.models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='./chatbot2_sim/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot2_sim/utils/user_dic.tsv')

intent = IntentModel(model_name='./chatbot2_sim/models/intent/intent_model.h5', preprocess=p)
query = "원주대 중앙동아리에는 뭐가 있어?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)
