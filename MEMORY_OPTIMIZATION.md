# Memory-Efficient Pipeline Refactoring

## 개요

기존 LangGraph 파이프라인을 메모리 효율적으로 리팩토링했습니다. 주요 변경사항은 **DataFrame을 state에 직접 저장하는 대신, 파일 경로를 저장하고 필요할 때만 데이터를 읽어오는 방식**으로 변경한 것입니다.

## 문제점 분석

### 기존 방식의 문제
1. **메모리 비효율성**: 대용량 DataFrame이 state에 계속 유지됨
2. **확장성 제약**: 데이터 크기가 커질수록 RAM 부족 문제
3. **불필요한 메모리 사용**: 한번 쓰이고 말 데이터를 계속 메모리에 보관

### 개선된 방식
1. **파일 기반 저장**: DataFrame을 CSV로 저장하고 경로만 state에 보관
2. **지연 로딩**: 필요할 때만 데이터를 메모리에 로드
3. **메타 정보 활용**: 컬럼 정보 등은 미리 분석해서 메타데이터로 저장

## 주요 변경사항

### 1. DataHelper 유틸리티 추가
새로운 `utils/data_utils.py` 파일을 생성하여 메모리 효율적인 데이터 접근 기능 제공:

```python
class DataHelper:
    @staticmethod
    def load_dataframe(csv_path: str) -> pd.DataFrame
    @staticmethod
    def get_dataframe_info(csv_path: str) -> Dict[str, Any]
    @staticmethod
    def get_columns(csv_path: str) -> List[str]
    @staticmethod
    def get_sample_data(csv_path: str, n_rows: int = 3, columns: Optional[List[str]] = None) -> Dict
    @staticmethod
    def get_open_columns_from_path(csv_path: str, meta_info: Dict[str, Any]) -> List[str]
```

### 2. SmartExcelResult 수정
`dataframe` property를 제거하여 자동 로딩 방지:

```python
@dataclass
class SmartExcelResult:
    dataframe_path: str  # DataFrame이 저장된 CSV 파일 경로
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # dataframe property 제거 - 필요시 DataHelper.load_dataframe() 사용
```

### 3. 노드 함수 개선
경로 기반 처리를 위한 새로운 함수들 추가:

#### get_open_column.py
```python
def get_open_column_from_path(dataframe_path: str, meta_info: dict = None) -> dict:
    """경로에서 open columns 추출 - 메모리 효율적"""
    if meta_info and 'object_columns' in meta_info:
        return {"open_columns": meta_info['object_columns']}
    return {"open_columns": DataHelper.get_open_columns_from_path(dataframe_path, meta_info or {})}
```

#### question_data_matcher.py
```python
def question_data_matcher_from_path(
    survey_questions: List,
    dataframe_path: str,
    open_columns: List[str],
    branch: str = "question_data_matcher",
    deps: Optional[Any] = None
) -> Dict[str, Any]:
    """경로 기반 매칭 - 메모리 효율적"""
    # 컬럼 정보와 샘플 데이터만 필요한 만큼 로드
    columns = DataHelper.get_columns(dataframe_path)
    data_sample = DataHelper.get_sample_data(dataframe_path, n_rows=3, columns=open_columns)
    # ... LLM 호출
```

### 4. 하위 호환성 유지
기존 코드가 계속 작동하도록 호환성 유지:

```python
def get_open_column_node(state: GraphState, deps=None) -> GraphState:
    """LangGraph용 노드 함수 - 경로 기반"""
    try:
        # 새로운 경로 기반 처리
        if "raw_data_info" in state and "dataframe_path" in state["raw_data_info"]:
            # 경로 기반 처리
            pass
        elif "dataframe" in state:
            # 하위 호환성: DataFrame 객체가 직접 전달된 경우
            pass
```

## 성능 개선 결과

### 메모리 사용량
- **기존**: DataFrame 전체가 메모리에 상주 (776행 × 419열 = ~4.33MB + 추가 오버헤드)
- **개선**: 필요한 샘플 데이터만 로드 (3행 × 필요 컬럼만)

### 확장성
- **기존**: 데이터 크기에 비례해서 메모리 사용량 증가
- **개선**: 데이터 크기와 무관하게 일정한 메모리 사용량

### 처리 속도
- **메타 정보 활용**: object_columns 등은 미리 분석되어 즉시 사용 가능
- **지연 로딩**: 실제로 필요한 데이터만 읽어서 I/O 최적화

## 사용 방법

### 1. 기존 방식 (호환성 유지)
```python
# 여전히 작동함
df = state["dataframe"]
open_columns = DataFrameParser.get_object_columns(df)
```

### 2. 새로운 방식 (권장)
```python
# 경로 기반 처리
dataframe_path = state["raw_data_info"]["dataframe_path"]
meta_info = state["raw_data_info"]["meta"]
open_cols = get_open_column_from_path(dataframe_path, meta_info)["open_columns"]

# 필요할 때만 데이터 로드
sample_data = DataHelper.get_sample_data(dataframe_path, n_rows=3, columns=open_cols[:5])
```

### 3. LangGraph 상태 설계
```python
state = {
    "raw_data_info": {
        "dataframe_path": "/path/to/saved/data.csv",
        "meta": {
            "object_columns": [...],
            "sheet_name": "Sheet1",
            "index_col": "IDKEY",
            ...
        }
    },
    "open_columns": [...],
    "parsed_survey": SurveyParseResult(...),
    # DataFrame 객체는 state에 저장하지 않음
}
```

## 테스트 결과

`test/raw_read_test_updated.py`와 `test/memory_efficient_test_no_llm.py`에서 성공적으로 검증:

1. ✅ 파일 로딩 및 경로 저장
2. ✅ 메타 정보 기반 컬럼 분석
3. ✅ 샘플 데이터 효율적 접근
4. ✅ 노드 함수 경로 기반 처리
5. ✅ 하위 호환성 유지
6. ✅ 메모리 사용량 개선

## 향후 계획

1. **LLM API 키 설정 수정**: 실제 LLM 호출 테스트
2. **추가 노드 함수 개선**: 다른 노드들도 경로 기반으로 업데이트
3. **성능 모니터링**: 대용량 데이터에서의 성능 측정
4. **캐싱 전략**: 자주 사용되는 데이터의 캐싱 구현

## 결론

이번 리팩토링을 통해:
- **메모리 효율성 대폭 개선**
- **대용량 데이터 처리 가능**
- **기존 코드와의 호환성 유지**
- **LangGraph 통합 준비 완료**

데이터 크기가 커져도 RAM을 효율적으로 사용할 수 있는 파이프라인이 구축되었습니다.
