import torch
from sentence_transformers import util
import numpy as np
import pandas as pd

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
    
    def _make_query_3(self, intent_name, tagged_text) :
        if intent_name == '종류':
            sql = " SELECT * FROM club_introduce_club WHERE club_name = '{}'".format(tagged_text)

        elif intent_name == '소개' :
            sql = " SELECT * FROM club_introduce_club WHERE club_name = '{}'".format(tagged_text)

        else :
            sql = " SELECT * FROM club_introduce_clubdetail WHERE club_id = '{}'".format(tagged_text)
        
        return sql

    # NER 태그를 실제 입력된 단어로 변환
    def tag_to_word(self, ner_predicts):
        for word, tag in ner_predicts:
                # 변환해야하는 태그가 있는 경우 추가
            if tag == 'B_OG':
                tagged_text = word
                break
            else :
                tagged_text = "NONE"
        return tagged_text
    
    
    # 답변 검색
    def search_1(self, intent_name):
        # 의도명으로 답변 검색
        sql = self._make_query_1(intent_name)
        answer = self.db.select_one(sql)

        return (answer['answer'], answer['answer_image'])

    # 답변 검색
    def search_2(self, intent_name, embedding_data):
        # 유사도 분석 데이터
        sim_data = torch.load('./chatbot3_last/models/sim/SBERT_embedding_Data.pt')

        cos_sim = util.cos_sim(embedding_data, sim_data)
        best_sim_idx = int(np.argmax(cos_sim)) # cos_sim의 최대값의 인덱스 반환
        best_sim = cos_sim[0, best_sim_idx].item()  # item()을 사용하여 Python float으로 변환
        sql = self._make_query_2(best_sim_idx+1)
        answer = self.db.select_one(sql)
        #print(answer, type(answer))
        print(best_sim)
        print(answer['answer'], answer['answer_image'])
        
        if answer['intent'] == intent_name :
            #answer = df['답변(Answer)'][best_sim_idx]
            #imageUrl = df['답변 이미지'][best_sim_idx]
            print("True")
            return (answer['answer'], answer['answer_image'])
            #return (answer, imageUrl)
            

        else :
            print("False")
            answer_text = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부 할게요."
            answer_image = None

            return (answer_text, answer_image)

    def search_3(self, intent_name, tagged_text, embedding_data) :
        # 개체명으로 백엔드 DB 검색
        sql = self._make_query_3(intent_name, tagged_text)
        try :
            club_info = self.db.select_one(sql)

        except Exception as e:
            club_info = None

        if club_info != None:
            if intent_name == "소개" and club_info['introducation'] != None:
                answer_text =  "동아리 {}에 대한 소개입니다: {}".format(club_info['club_name'], club_info['introducation'])
                if club_info['logo'] != None :
                    answer_image = club_info['logo']
                else :
                    answer_image = None
                return (answer_text, answer_image)
        
            elif intent_name == "종류" :
                answer_text = "{} 동아리는 다음과 같습니다.: {}".format(club_info['type'], club_info['club_name'])
                answer_image = None
                return (answer_text, answer_image)
            
            
            elif intent_name == "가입방법" and club_info['join'] != None:
                answer_text = "동아리 {}의 가입 방법은: {}".format(club_info['club_id'], club_info['join'])
                answer_image = None
                return (answer_text, answer_image)
            
            elif intent_name == "위치" and club_info['location'] != None :
                answer_text = "동아리 {}의 위치는: {}".format(club_info['club_id'], club_info['location'])
                answer_image = None
                return (answer_text, answer_image)

            elif intent_name == "활동" and club_info['activity'] != None :
                answer_text = "동아리 {}의 주요 활동은: {}".format(club_info['club_id'], club_info['activity'])
                answer_image = None
                return (answer_text, answer_image)

            elif intent_name == "회비" and club_info['fee'] != None:
                answer_text = "동아리 {}의 회비는: {}".format(club_info['club_id'], club_info['fee'])
                answer_image = None
                return (answer_text, answer_image)
        
            else:
                # 유사도 분석 데이터
                sim_data = torch.load('./chatbot3_last/models/sim/SBERT_embedding_Data.pt')

                cos_sim = util.cos_sim(embedding_data, sim_data)
                best_sim_idx = int(np.argmax(cos_sim)) # cos_sim의 최대값의 인덱스 반환
                best_sim = cos_sim[0, best_sim_idx].item()  # item()을 사용하여 Python float으로 변환
                sql = self._make_query_2(best_sim_idx+1)
                answer = self.db.select_one(sql)
                #print(answer, type(answer))
                print(best_sim)
                print(answer['answer'], answer['answer_image'])
                
                if answer['intent'] == intent_name :
                    #answer = df['답변(Answer)'][best_sim_idx]
                    #imageUrl = df['답변 이미지'][best_sim_idx]
                    print("True")
                    answer_text = answer['answer'] + '\n이 답변은 업데이트 되지 않은 답변이니 자세한 사항은 동아리로 직접 문의 하시기 바랍니다.'
                    return (answer_text, answer['answer_image'])
                    #return (answer, imageUrl)
                    

                else :
                    print("False")
                    answer_text = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부 할게요."
                    answer_image = None

                    return (answer_text, answer_image)
               
        else :
            return (None, None) 