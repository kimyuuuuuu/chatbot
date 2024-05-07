import pandas as pd
from tqdm import tqdm
import numpy as np
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer
from chatbot2_sim.utils.Preprocess import Preprocess

# 데이터 읽어오기
train_file = "./chatbot2_sim/models/sim/train_data_5.xlsx"
data = pd.read_excel(train_file)
queries = data['질문(Query)'].tolist() 

p = Preprocess(word2index_dic='./chatbot2_sim/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot2_sim/utils/user_dic.tsv')

# 단어 시퀀스 생성
keywords_list = []
for sentence in queries:
  pos = p.pos(sentence)
  keywords = p.get_keywords(pos, without_tag=True)
  keyword_string = ' '.join(keywords)  
  keywords_list.append(keyword_string)

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

vectors = model.encode(keywords_list)
vector_df = pd.DataFrame(vectors)
vector_df.to_excel("./chatbot2_sim/models/sim/train_data_SBERT.xlsx", index=False)

embedding_data = torch.tensor(vectors.tolist())
torch.save(embedding_data, './chatbot2_sim/models/sim/SBERT_embedding_Data.pt')