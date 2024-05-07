import torch
from sentence_transformers import util
import numpy as np

class FindAnswer:
    def __init__(self, db):
        self.db = db
        # 미리 학습된 SBERT 임베딩 데이터 로드
        self.sim_data = torch.load('chatbot2_sim/models/sim/SBERT_embedding_Data.pt')
        
        # 해당 임베딩 데이터에 대한 매핑 정보를 로드하여 DataFrame으로 읽음
        self.data_mapping = pd.read_excel('chatbot2_sim/models/sim/train_data_SBERT.xlsx')

    # 검색 쿼리 생성
    def _make_query(self, intent_name, ner_tags):
        sql = "select * from chatbot_train_data"
        if intent_name != None and ner_tags == None:
            sql = sql + " where intent='{}' ".format(intent_name)

        elif intent_name != None and ner_tags != None:
            where = ' where intent="%s" ' % intent_name
            if (len(ner_tags) > 0):
                where += 'and ('
                for ne in ner_tags:
                    where += " ner like '%{}%' or ".format(ne)
                where = where[:-3] + ')'
            sql = sql + where

        # 동일한 답변이 2개 이상인 경우, 랜덤으로 선택
        sql = sql + " order by rand() limit 1"
        return sql


    # 답변 검색
    def search(self, intent_name, embedding_data):
        # 유사도 분석 데이터
        sim_data = torch.load('chatbot2_sim/models/sim/SBERT_embedding_Data.pt')

        cos_sim = util.cos_sim(embedding_data, sim_data)
        best_sim_idx = int(np.argmax(cos_sim)) # cos_sim의 최대값의 인덱스 반환
        

        if self.df
            

        return (answer['answer'], answer['answer_image'])

            

        return (answer['answer'], answer['answer_image'])