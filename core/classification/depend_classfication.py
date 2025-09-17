
# if question type is dependent:
# 데이터는 문장 형태, related_Question에서의 값 기반하여 해석된 값을 추가필요
# 각 문장은 S,V,C로 분해 하여 키워드 추출
# 각 문장은 추출된 S,V,C 기반, 원자적 의미를 가지도록 최대 3개의 문장을 생성
# 각 문장을 embedding 처리
# embedding 한 결과 KNN -> CSLS -> MCL 기반 mapping 진행
# mapping 한 결과를 key로 묶음
# key 기반하여 유의미한 매핑이 있는지 다시 확인
# LLM 확인하여 동일 의미 또는 오탈자로 인한 group 있는지 확인
# LLM 활용하여 S,V 기반 요약된 key 에 대한 요약 문장 생성
# 매핑 테이블 생성

# if question type is sentence:
# # 각 문장은 S,V,C로 분해 하여 키워드 추출
# 각 문장은 추출된 S,V,C 기반, 원자적 의미를 가지도록 최대 3개의 문장을 생성
# 각 문장을 embedding 처리
# embedding 한 결과 KNN -> CSLS -> MCL 기반 mapping 진행
# mapping 한 결과를 key로 묶음
# key 기반하여 유의미한 매핑이 있는지 다시 확인
# LLM 확인하여 동일 의미 또는 오탈자로 인한 group 있는지 확인
# LLM 활용하여 S,V 기반 요약된 key 에 대한 요약 문장 생성
# 매핑 테이블 생성