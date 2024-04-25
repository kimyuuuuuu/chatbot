import chardet

# 데이터 인코딩 확인
train_file = "./models/intent/total_train_data_4.csv"

with open(train_file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(10000))

print(result)
