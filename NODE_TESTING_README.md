# 🧪 Graph Node Unit Testing Suite

## 📋 Overview

이 디렉토리에는 LangGraph 파이프라인의 모든 노드들을 개별적으로 테스트하는 포괄적인 유닛테스트 코드들이 포함되어 있습니다.

## 🎯 Testing Philosophy

각 노드를 **개별 단위(unit)**로 테스트하여:
- 노드 간 의존성 최소화
- 개별 노드의 입출력 검증
- 모킹(mocking)을 통한 외부 의존성 격리
- 엣지 케이스 및 오류 상황 처리 검증

## 📁 Test Files Structure

### 🔧 Core Test Files

1. **`test_comprehensive_nodes.py`** - 전체 노드 포괄 테스트
   - 모든 노드의 기본 기능 테스트
   - 통합 플로우 테스트
   - 데이터 흐름 검증

2. **`test_stage1_nodes_detailed.py`** - Stage 1 노드 세부 테스트
   - Survey Loader Node
   - Data Loader Node  
   - Survey Parser Node
   - Column Extractor Node
   - Question Matcher Node

3. **`test_stage2_nodes_detailed.py`** - Stage 2 노드 세부 테스트
   - Stage2 Main Node
   - Stage2 Word Node
   - Stage2 Sentence Node
   - Stage2 ETC Node
   - Stage2 Router Tests

4. **`test_individual_nodes_pytest.py`** - Pytest 스타일 테스트
   - Pytest fixtures 활용
   - Parametrized testing
   - 고급 테스트 패턴

5. **`test_nodes_fixed.py`** - 수정된 노드 테스트
   - 실제 발견된 이슈들 수정
   - 실용적인 테스트 접근법

### 🚀 Test Runner

6. **`run_all_node_tests.py`** - 마스터 테스트 러너
   - 모든 테스트 자동 실행
   - 단계별 테스트 진행
   - 종합 결과 리포트

## 🎮 How to Run Tests

### 전체 테스트 실행
```bash
cd /home/cyyoon/test_area/ai_text_classification/2.langgraph
python run_all_node_tests.py
```

### 개별 테스트 파일 실행
```bash
# Stage 1 노드만 테스트
python test_stage1_nodes_detailed.py

# Stage 2 노드만 테스트  
python test_stage2_nodes_detailed.py

# 수정된 테스트만 실행
python test_nodes_fixed.py

# Pytest 스타일 (pytest 설치된 경우)
pytest test_individual_nodes_pytest.py -v
```

## 🧬 Tested Nodes

### 📊 Stage 1 - Data Preparation Nodes

| Node | Function | Test Coverage |
|------|----------|---------------|
| `survey_loader` | 설문 파일 로딩 | ✅ 파일 읽기, 오류 처리 |
| `data_loader` | 데이터 파일 로딩 | ✅ CSV/Excel 지원, 형식 검증 |
| `survey_parser` | 설문 파싱 | ✅ LLM 모킹, 질문 유형 분류 |
| `survey_context` | 컨텍스트 추출 | ✅ LLM 응답 처리 |
| `column_extractor` | 개방형 컬럼 추출 | ✅ 질문 유형별 필터링 |
| `question_matcher` | 질문-데이터 매칭 | ✅ 매칭 로직, 신뢰도 검증 |

### 🔄 Stage 2 - Data Processing Nodes

| Node | Function | Test Coverage |
|------|----------|---------------|
| `stage2_main` | 전처리 메인 | ✅ 큐 관리, 데이터 샘플링 |
| `stage2_word_node` | 단어 분석 | ✅ LLM 모킹, 키워드 추출 |
| `stage2_sentence_node` | 문장 분석 | ✅ 문법 체크, 감정 분석 |
| `stage2_etc_node` | 기타 처리 | ✅ 예외 상황 처리 |
| `stage2_next_question` | 질문 순회 | ✅ 인덱스 관리, 완료 조건 |

### 🔀 Router & Shared Nodes

| Component | Function | Test Coverage |
|-----------|----------|---------------|
| `stage2_type_router` | 질문 유형 라우팅 | ✅ 모든 유형, 예외 처리 |
| `stage2_completion_router` | 완료 조건 라우팅 | ✅ 계속/종료 조건 |
| `survey_data_integrate` | 데이터 통합 | ✅ 큐 생성, 매칭 통합 |
| `memory_status_check` | 메모리 상태 체크 | ✅ 상태 모니터링 |
| `stage_tracker` | 단계 추적 | ✅ 상태 전환, 로깅 |

## 🎭 Mocking Strategy

### LLM Client Mocking
```python
@patch('io_layer.llm.client.LLMClient')
def test_with_llm_mock(mock_llm_client):
    mock_instance = Mock()
    mock_instance.chat.return_value = ({"parsed": {...}}, Mock())
    mock_llm_client.return_value = mock_instance
```

### Prompt Resolver Mocking
```python
@patch('config.prompt.prompt_loader.resolve_branch')
def test_with_prompt_mock(mock_resolve_branch):
    mock_resolve_branch.return_value = {
        'system': 'Mock system prompt',
        'user_template': 'Mock template: {data}',
        'schema': Mock()
    }
```

## 🔍 Test Data Fixtures

### Survey Data
```python
def create_mock_survey_data():
    return """
    Q1. 브랜드에 대한 전반적인 만족도는 어떠신가요?
    ① 매우 만족 ② 만족 ③ 보통 ④ 불만족 ⑤ 매우 불만족
    
    Q2. 제품 품질에 대해서는 어떻게 생각하시나요?
    """
```

### Response Data
```python
def create_mock_data_csv():
    return pd.DataFrame({
        'Q1': ['매우 만족', '만족', '보통'],
        'Q2': ['품질이 정말 좋아요', '괜찮은 편입니다', '그냥 그래요'],
        'respondent_id': [1, 2, 3]
    })
```

## 📈 Test Results Summary

### ✅ Working Tests
- **Basic Node Structure**: 모든 노드가 올바른 시그니처 보유
- **Survey Loader**: 파일 읽기 및 기본 처리
- **Data Loader**: CSV/Excel 파일 로딩
- **Stage2 Main**: 큐 관리 및 데이터 전처리
- **Routers**: 조건부 라우팅 로직
- **Stage Trackers**: 상태 전환 관리

### ⚠️ Areas for Improvement  
- **LLM Integration**: 실제 LLM 호출 대신 모킹 필요
- **State Management**: 노드 간 상태 전달 표준화
- **Error Handling**: 예외 상황 처리 강화
- **Performance**: 대용량 데이터 처리 테스트

## 🚀 Next Steps

### 1. 테스트 확장
- 더 많은 엣지 케이스 추가
- 성능 테스트 (대용량 데이터)
- 메모리 사용량 모니터링

### 2. 통합 테스트
- 전체 파이프라인 end-to-end 테스트
- 실제 데이터로 검증
- 병목 지점 식별

### 3. 자동화
- CI/CD 파이프라인 통합
- 자동 회귀 테스트
- 성능 벤치마크 추적

## 💡 Testing Best Practices

### ✅ Do's
- 각 노드를 독립적으로 테스트
- 외부 의존성은 모킹 사용
- 다양한 입력 조건 테스트
- 명확한 assertion 작성

### ❌ Don'ts  
- 실제 LLM API 호출 (비용, 속도)
- 노드 간 강한 결합 테스트
- 하드코딩된 파일 경로 사용
- 환경 의존적 테스트

## 🎯 Conclusion

이 테스트 스위트는 LangGraph 파이프라인의 **안정성**과 **신뢰성**을 보장하기 위한 포괄적인 검증 도구입니다. 각 노드의 개별 기능을 철저히 테스트함으로써 전체 시스템의 품질을 향상시킵니다.

---

*테스트는 코드의 품질을 보장하는 투자입니다. 🛡️*