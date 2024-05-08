import threading
import json

from chatbot2_sim.config.DatabaseConfig import *
from chatbot2_sim.utils.Database import Database
from chatbot2_sim.utils.BotServer import BotServer
from chatbot2_sim.utils.Preprocess import Preprocess
from chatbot2_sim.models.intent.IntentModel import IntentModel
from chatbot2_sim.models.sim.SimModel import SimModel
from chatbot2_sim.utils.FindAnswer import FindAnswer


# 전처리 객체 생성
p = Preprocess(word2index_dic='chatbot2_sim/train_tools/dict/chatbot_dict.bin',
               userdic='chatbot2_sim/utils/user_dic.tsv')

# 의도 파악 모델
intent = IntentModel(model_name='chatbot2_sim/models/intent/intent_model.h5', preprocess=p)

# 유사도 분석 모델
sim = SimModel(preprocess=p)


def to_client(conn, addr, params):
    db = params['db']

    try:
        db.connect()  # 디비 연결

        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        print('===========================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)


        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]
        print(intent_name)

        # 질문 임베딩
        embedding_data = sim.create_pt(query)

        # 답변 검색
        f = FindAnswer(db)
        if f != None :
            if intent_name == '인사' or intent_name == '욕설' or intent_name == '기타' :
                answer_text, answer_image = f.search_1(intent_name)

            else : 
                answer_text, answer_image = f.search_2(intent_name, embedding_data)
        

        send_json_data_str = {
            "Query" : query,
            "Answer": answer_text,
            "AnswerImageUrl" : answer_image,
            "Intent": intent_name
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())

    except Exception as ex:
        print(ex)

    finally:
        if db is not None: # db 연결 끊기
            db.close()
        conn.close()


if __name__ == '__main__':

    # 질문/답변 학습 디비 연결 객체 생성
    db = Database(
        host=DB_HOST, user=DB_USER, db_name=DB_NAME
    )
    print("DB 접속")

    port = 5050
    listen = 100

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            "db": db
        }

        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
            params
        ))
        client.start()