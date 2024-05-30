from chatbot1_ner.utils.Preprocess import Preprocess
from chatbot1_ner.models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='C:/chatbot/chatbot1_ner/train_tools/dict/chatbot_dict.bin',
               userdic='C:/chatbot/chatbot1_ner/utils/user_dic.tsv')

ner = NerModel(model_name='./chatbot1_ner/models/ner/ner_model_testNER3.h5', proprocess=p)
query = ['프레이즈에 대해 알려줘','CCC에 대해 알려줘','중앙동아리에 대해 알려줘','취동에 대해 알려줘','학습동아리에 대해 알려줘','스핀에 대해 알려줘','Tagthebills에 대해 알려줘']

for i in query :
  predicts = ner.predict(i)
  tags = ner.predict_tags(i)
  print('predicts:',predicts)
  print('tags',tags)
