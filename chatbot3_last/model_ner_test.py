from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='C:/chatbot/chatbot1_ner/train_tools/dict/chatbot_dict.bin',
               userdic='C:/chatbot/chatbot1_ner/utils/user_dic.tsv')


ner = NerModel(model_name='C:/chatbot/chatbot1_ner/models/ner/ner_model_testNER2.h5', proprocess=p)
query = ['학습동아리에 대해 알려줘']

for i in query :
  predicts = ner.predict(i)
  tags = ner.predict_tags(i)

  tagged_text = [word for word, tag in predicts if tag in tags]

  print(f'Query: {query}')
  print('Predicts:', predicts)
  print('Tags:', tags)
  print('Tagged Text:', tagged_text)
  print() 