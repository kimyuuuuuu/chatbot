from konlpy.tag import Komoran # 코모란 불러오기(토큰화 모듈)
import pickle # 딕셔너리, 리스트 등의 자료형을 변환 없이 파일로 저장, 불러오는 모듈

class Preprocess :
  def __init__(self, userdic=None): # 생성자
    # 형태소 분석기 초기화
    self.komoran = Komoran(userdic=userdic)
    # 제외할 품사
    # 참조 : https://docs.komoran.kr/firststep/postypes.html
    self.exclusion_tags = [
      "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", # 관계언 제거
			"SF", "SP", "SS", "SE", "SO", # 기호 제거
			"EP", "EF", "EC", "ETN", "ETM", # 어미 제거
			"XSN", "XSV", "XSA", # 접미사 제거 
    ]

  # 형태소 분석기 POS 태거
  def pos(self, sentence) :
    return self.komoran.pos(sentence)
  
  # 불용어 제거 후 필요한 품사 정보만 가져오기 
  def get_keywords(self, pos, without_tag=False):
    f = lambda x: x in self.exclusion_tags
    word_list = []
    for p in pos :
      if f(p[1]) is False : # p의 결과 = ('단어', '품사') <- p[0] = 단어, p[1] = 품사
        word_list.append(p if without_tag is False else p[0])
    return word_list