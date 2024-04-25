import pandas as pd

# 데이터 읽어오기
train_file = "./models/intent/total_train_data_3.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

