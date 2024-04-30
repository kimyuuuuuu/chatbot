import os
import pandas as pd
import json

purpose = './raw_data/Purpose_conversation'


# 폴더별로 파일 불러와서 csv화
p_files = os.listdir(purpose) # purpose의 폴더 이름 불러옴 (TL~)

for pidx, fname in enumerate(p_files) :
  tpath = purpose + '/' + fname
  TL = os.listdir(tpath) # TL하나의 폴더이름 저장
  
  for pidx2, fname2 in enumerate(TL) :
    print(fname2)
    cpath = tpath + '/' + fname2 
    category = os.listdir(cpath) # TL 속 01~ 파일 이름 저장
    
    p_extracted_data = []
    for pidx3, fname3 in enumerate(category) :
      # print(fname3)
      fpath = cpath + '/' + fname3
      try:
        with open(fpath, "r") as file:
          data = json.load(file)
          # print(file)
      except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} in {fpath}")
        continue

      for info_lines in data['info'] :
        lines = info_lines['annotations']['lines']
        for line in lines:
          norm_text = line['norm_text']
          text = (norm_text[2:]).lstrip()
          p_extracted_data.append(text)

# 추출한 데이터를 데이터프레임으로 변환
df = pd.DataFrame(data=p_extracted_data, columns=['text'])

# 데이터프레임을 CSV 파일로 저장
csv_file_path = './data/p_extracted_data.csv'
df.to_csv(csv_file_path, index=False)

print(f'Data saved to {csv_file_path}')

print(p_files)
