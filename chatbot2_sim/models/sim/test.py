import torch
import pandas as pd

loaded_vectors = torch.load('./chatbot2_sim/models/sim/SBERT_embedding_Data.pt')
print(loaded_vectors)

df_vectors = pd.read_excel("./chatbot2_sim/models/sim/train_data_SBERT.xlsx")
print(df_vectors.head())
