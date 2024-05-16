class save_conversation:
  def __init__(self, db):
    self.db = db

  def save_conv(self, query, answer, intent):
    sql = "INSERT INTO save_conversation (query, answer, intent) VALUES (%s, %s, %s)"

    # 튜플 형식으로 값 바인딩
    values = (query, answer, intent)

    # 커서 객체를 사용하여 쿼리 실행
    with self.conn.cursor() as cursor:
        cursor.execute(sql, values)
        self.conn.commit()