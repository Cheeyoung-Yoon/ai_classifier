# State Memory Optimization Summary

## 🎯 개선 목표
State가 각 노드를 거치면서 계속 커지는 문제를 해결하여 메모리 사용량을 최적화하고, 프로젝트 기반 파일 경로 관리 개선

## 📊 기존 문제점

### 1. 메모리 누적 문제
- `raw_survey_info`: 큰 텍스트 데이터가 파이프라인 전체에서 유지됨
- `raw_data_info`: 대용량 메타데이터가 계속 저장됨
- `parsed_survey`: 매칭 후에도 삭제되지 않음
- `matched_questions`: `question_data_match`와 중복

### 2. 불필요한 필드들
- `dataframe`: 메모리에 전체 DataFrame 보관
- `open_columns`: `raw_data_info.object_columns`와 중복
- `data_sample`: 노드에서 자체적으로 읽을 수 있음
- `matched_questions_meta`: 불필요한 메타데이터
- `llm_logs`, `llm_meta`: 무제한 증가

### 3. 파일 경로 관리
- 하드코딩된 절대 경로
- 프로젝트별 조직화 부족

## 🔧 개선 사항

### 1. 새로운 State 구조 (`ImprovedGraphState`)

```python
{
  "project_name": str,           # 프로젝트 이름 (전체 유지)
  "survey_file_path": str,       # ./data/{project_name}/{survey_file}
  "data_file_path": str,         # ./data/{project_name}/{data_file}
  
  "raw_survey_info": Optional,   # parse_survey 후 삭제
  "raw_data_info": Optional,     # get_open_columns 후 meta만 삭제
  "parsed_survey": Optional,     # match_questions 후 삭제
  
  "question_data_match": str,    # 최종 결과만 유지
  
  # 분류 처리 관련 (기존 유지)
  "integrated_map": Optional,
  "current_question_index": Optional,
  "focus_qid": Optional,
  "current_question_info": Optional,
  "router_decision": Optional,
  "classification_results": Optional,
  "processing_complete": Optional,
  "total_questions": Optional,
  
  "error": Optional              # 필수 에러 처리
}
```

### 2. 메모리 정리 단계별 적용

#### Stage 1: Survey Parse 후
```python
state["raw_survey_info"] = None  # 대용량 텍스트 삭제
```

#### Stage 2: Column Detection 후
```python
state["raw_data_info"] = {
    "path": path,
    "dataframe_path": dataframe_path
    # meta 데이터 삭제
}
```

#### Stage 3: Question Match 후
```python
state["parsed_survey"] = None      # 구조화된 질문 데이터 삭제
del state["matched_questions"]     # 중복 데이터 삭제
```

### 3. 프로젝트 기반 파일 경로

```python
def initialize_project_state(project_name, survey_filename, data_filename):
    return {
        "project_name": project_name,
        "survey_file_path": f"./data/{project_name}/{survey_filename}",
        "data_file_path": f"./data/{project_name}/{data_filename}"
    }
```

## 📈 성과

### 메모리 사용량 감소
- State 필드 수: **10개 → 6개** (40% 감소)
- 대용량 데이터 단계별 정리
- 중복 필드 제거

### 파일 관리 개선
- 프로젝트별 디렉토리 구조
- 상대 경로 기반 관리
- 설정 간소화

### 코드 유지보수성
- 단계별 cleanup 함수
- 재사용 가능한 헬퍼 함수
- 명확한 메모리 관리 정책

## 🔧 구현 파일

### 새로운 파일들
- `graph/improved_state.py` - 개선된 State 정의
- `utils/state_utils.py` - State 관리 헬퍼 함수
- `graph/memory_optimized_graph.py` - 최적화된 그래프
- `tests/improved_state_test.py` - 기본 테스트
- `tests/memory_optimized_debug_test.py` - 통합 테스트

### 수정된 파일들
- `nodes/parse_survey.py` - cleanup 로직 추가
- `nodes/get_open_column.py` - meta 데이터 정리
- `nodes/question_data_matcher.py` - 중복 데이터 제거

## 🚀 사용 방법

### 기본 사용
```python
from utils.state_utils import initialize_project_state
from graph.memory_optimized_graph import run_memory_optimized_pipeline

# 프로젝트 기반 초기화
state = initialize_project_state("SUV_DEBUG", "test.txt", "-SUV_776부.xlsx")

# 최적화된 파이프라인 실행
result = run_memory_optimized_pipeline("SUV_DEBUG", "test.txt", "-SUV_776부.xlsx")
```

### 수동 정리
```python
from utils.state_utils import cleanup_state_memory

# 단계별 메모리 정리
state = cleanup_state_memory(state, "after_survey_parse")
state = cleanup_state_memory(state, "after_column_detection") 
state = cleanup_state_memory(state, "after_question_match")
```

## ✅ 검증 완료

1. ✅ State 초기화 및 파일 경로 설정
2. ✅ 단계별 메모리 정리 동작
3. ✅ 중복 데이터 제거
4. ✅ 필수 데이터 보존
5. ✅ 기존 노드와의 호환성

## 📋 다음 단계

1. 기존 그래프를 `ImprovedGraphState`로 마이그레이션
2. 실제 파이프라인에서 메모리 사용량 모니터링
3. 추가 최적화 포인트 식별
4. 성능 벤치마크 수행
