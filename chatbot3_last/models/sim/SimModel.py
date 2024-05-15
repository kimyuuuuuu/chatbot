
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 임베딩 모델 모듈
class SimModel:
  def __init__(self, preprocess):
    self.model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    self.p = preprocess

  # 질문 임베딩
  def create_pt(self, query) :
    # 질문 전처리
    pos = self.p.pos(query)
    keywords = self.p.get_keywords(pos, without_tag=True)

    # 질문 하나의 문장으로 만들기
    keyword_string = ' '.join(keywords)  

    # 모델 임베딩
    vectors_encode = self.model.encode(keyword_string)
    vectors = torch.tensor(vectors_encode)
    
    return vectors
  