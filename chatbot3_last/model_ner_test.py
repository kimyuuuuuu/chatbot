from chatbot3_last.utils_list.Preprocess import Preprocess
from chatbot3_last.models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='./chatbot3_last/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot3_last/utils/user_dic.tsv')


ner = NerModel(model_name='./chatbot3_last/models/ner/ner_model_testNER2.h5', proprocess=p)
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