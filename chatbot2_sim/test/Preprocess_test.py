from chatbot2_sim.utils.Preprocess import Preprocess

sent = "루시우스 가입 방법 알려줘"

# 전처리 객체 생성
p = Preprocess(userdic='./chatbot2_sim/utils/user_dic.tsv')

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태그와 같이 키워드 출력
ret = p.get_keywords(pos, without_tag=False)
print(ret)

# 품사 태그 없이 키워드 출력
ret = p.get_keywords(pos, without_tag = True)
print(ret)