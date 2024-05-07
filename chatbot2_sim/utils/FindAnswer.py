import torch
from sentence_transformers import util
import numpy as np

class FindAnswer:
    def __init__(self, db):
        self.db = db

    
    # 검색 쿼리 생성
    def _make_query_1(self, intent_name):
        sql = "select * from chatbot_train_data"
        if intent_name != None :
            sql = sql + " where intent='{}' ".format(intent_name)

        # 동일한 답변이 2개 이상인 경우, 랜덤으로 선택
        sql = sql + " order by rand() limit 1"
        return sql


    # 검색 쿼리 생성
    def _make_query_2(self, query_id):
        sql = "SELECT * FROM chatbot_train_data"
        if query_id != None :
            sql = sql + " WHERE id = '{}' LIMIT 1".format(query_id)

        return sql
    
    # 답변 검색
    def search_1(self, intent_name):
        # 의도명으로 답변 검색
        sql = self._make_query_1(intent_name)
        answer = self.db.select_one(sql)

        return (answer['answer'], answer['answer_image'])


    # 답변 검색
    def search_2(self, intent_name, embedding_data):
        # 유사도 분석 데이터
        sim_data = torch.load('chatbot2_sim/models/sim/SBERT_embedding_Data.pt')

        cos_sim = util.cos_sim(embedding_data, sim_data)
        best_sim_idx = int(np.argmax(cos_sim)) # cos_sim의 최대값의 인덱스 반환
        best_sim = cos_sim[0, best_sim_idx].item()  # item()을 사용하여 Python float으로 변환
        sql = self._make_query_2(best_sim_idx)
        answer = self.db.select_one(sql)
        #print(answer, type(answer))
        print(best_sim)
        print(answer['answer'], answer['answer_image'])
        
        if answer['intent'] == intent_name :
            return (answer['answer'], answer['answer_image'])