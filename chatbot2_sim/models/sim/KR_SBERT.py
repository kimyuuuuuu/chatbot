import pandas as pd
from tqdm import tqdm
import numpy as np
tqdm.pandas()

import torch
from sentence_transformers import SentenceTransformer

def read_data(filename):
  data = pd.read_excel(filename)
  # 'query' 열에서 데이터 추출
  queries = data['질문(Query)'].tolist()
  return queries

train_file = "./chatbot2_sim/models/sim/train_data_5.xlsx"
df = read_data(train_file)
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

vectors = model.encode(df)
vector_df = pd.DataFrame(vectors)
vector_df.to_excel("./chatbot2_sim/models/sim/train_data_SBERT.xlsx", index=False)

embedding_data = torch.tensor(vectors.tolist())
torch.save(embedding_data, './chatbot2_sim/models/sim/SBERT_embedding_Data.pt')