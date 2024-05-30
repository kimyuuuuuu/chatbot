# import chardet

# # 데이터 인코딩 확인
# train_file = "./models/intent/total_train_data_4.csv"

# with open(train_file, 'rb') as rawdata:
#   result = chardet.detect(rawdata.read(10000))

# print(result)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

model = keras.models.load_model('./chatbot2_sim/models/intent/intent_model')

model.get_config()