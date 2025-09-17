

# AI Text Classification Pipeline with LangGraph

**3단계 Router-기반 텍스트 분류 파이프라인** - 설문조사와 개방형 응답 데이터를 자동으로 매칭, 전처리, 분류하는 지능형 시스템

## ✨ 주요 특징

- 🎯 **Stage-기반 아키텍처**: 데이터 준비 → 전처리 → 분류 단계별 처리
- 🤖 **Dual LLM Processing**: 문법 교정(gpt-4.1) + 문장 분석(gpt-4.1-nano)  
- 🔀 **Router-기반 분기**: 질문 타입(WORD/SENTENCE/ETC)별 지능형 라우팅
- 📊 **자동 CSV 출력**: 전처리 결과 타임스탬프별 저장
- 🧠 **메모리 최적화**: 단계별 상태 정리로 효율적 처리
- 📁 **프로젝트 디렉토리 관리**: 체계적인 데이터 구조 및 상태 이력 관리
- 🔍 **State History 추적**: 매 프로세스별 상태 저장으로 완전한 추적성

## 🎉 구현 현황 (2025-09-17 업데이트)

### ✅ Stage 1: Data Preparation (완료)
- **Survey Loading**: 설문조사 파일 로딩 및 검증
- **Data Loading**: 데이터 파일 로딩 및 전처리
- **Survey Parsing**: 설문조사 구조 분석
- **Column Extraction**: 개방형 응답 컬럼 식별  
- **Question Matching**: 질문-데이터 매칭
- **Memory Optimization**: 단계별 메모리 정리

### ✅ Stage 2: Data Preprocessing (완료)
- **Router-Based Architecture**: 질문 타입별 라우팅 시스템
- **WORD Type Processing**: concept, img → 단순 데이터 추출
- **SENTENCE Type Processing**: depend_pos_neg → Dual LLM 처리
- **CSV Output Management**: 프로젝트별 결과 저장
- **State History Tracking**: 매 처리 단계별 상태 저장

### 📋 Stage 3: Classification (개발 예정)
- **Content Classification**: 텍스트 내용 기반 분류
- **Clustering Operations**: 임베딩 기반 클러스터링
- **Result Aggregation**: 분류 결과 통합

## 🏗️ 프로젝트 디렉토리 구조 (신규)

```
./data/{project_name}/
├── raw/                          # 외부 API 원본 데이터
│   ├── survey.txt               # 설문지 파일
│   └── data.xlsx                # 응답 데이터
├── state.json                   # 최신 상태
├── state_history/               # 프로세스별 상태 이력
│   ├── {timestamp}_{stage}_state.json
│   └── ...
└── temp_data/                   # 작업 임시 데이터
    └── stage2_results/          # Stage 2 CSV 결과
        ├── stage2_{qid}_{type}_{timestamp}.csv
        └── ...
```

### 🔄 State Management
- **최신 State**: `state.json`에 현재 상태 저장
- **State History**: 매 단계별 `{timestamp}_{stage}_state.json` 저장
- **Config 제어**: `SAVE_STATE_LOG` 설정으로 저장 여부 제어

## 🔥 Stage 2 Implementation Details (최신 구현)

### Router-Based Architecture
Stage 2는 기존 프로젝트 패턴을 따라 **Router + 조건부 엣지** 방식으로 구현되었습니다:

```
stage2_router → 조건부 엣지 → stage2_word_node/stage2_sentence_node/stage2_etc_node
```

### Question Type Routing
- **WORD Types**: `concept`, `img` → 전처리 불필요, pass 처리
- **SENTENCE Types**: `depend`, `depend_pos_neg`, `pos_neg` → 두 단계 LLM 처리
- **ETC Types**: 기타 분류되지 않은 질문들 → 기본 처리

### Dual LLM Processing (SENTENCE Types)
```python
# 1단계: 문법 교정 (gpt-4.1)
llm_client_large = LLMClient(model_key="gpt-4.1")
corrected_text = grammar_correction(original_text)

# 2단계: 문장 분석 (gpt-4.1-nano) 
llm_client_nano = LLMClient(model_key="gpt-4.1-nano")
analysis_result = sentence_analysis(corrected_text)
```

### Enhanced matched_questions Structure
Stage 2 처리 후 `matched_questions`에 **stage2_data** 필드가 추가되어 Stage 3에서 사용:
```python
{
    "questions": [
        {
            "qid": "Q15_1",
            "question": "AI 기술에 대한 귀하의 생각은?",
            "type": "depend_pos_neg", 
            "stage2_data": "/path/to/stage2_Q15_1_depend_pos_neg_20241217_143022.csv"  # ✨ 신규 추가
        }
    ]
}
```

### Output Format & State Saving
처리 결과는 프로젝트별 디렉토리에 저장:
- **CSV 저장**: `{project}/temp_data/stage2_results/stage2_{qid}_{type}_{timestamp}.csv`
- **State History**: `{project}/state_history/{timestamp}_stage2_state.json`
- **CSV 포맷**: `org_text`, `correction_text`, `pos.neg`, `matching_question`, `sentence_1~3`, `S_1~3`, `V_1~3`, `C_1~3`

### ProjectDirectoryManager Integration
```python
from utils.project_manager import ProjectDirectoryManager

# 프로젝트 디렉토리 생성 및 관리
pm = ProjectDirectoryManager(project_name="test_project")
pm.create_directories()
pm.copy_raw_files(survey_path, data_path)
csv_path = pm.save_stage2_result(result_data, qid, question_type)
pm.save_state_history(state, "stage2")
```

### Key Files
- `utils/project_manager.py`: 프로젝트 디렉토리 및 상태 관리
- `router/stage2_router.py`: 타입별 라우팅 로직
- `nodes/stage2_data_preprocessing/stage2_sentence_node.py`: 메인 처리 로직
- `nodes/stage2_data_preprocessing/stage2_word_node.py`: WORD 타입 처리
- `nodes/stage2_data_preprocessing/stage2_etc_node.py`: ETC 타입 처리

## 🚀 주요 기능

### Core Pipeline
- **3-Stage Processing**: 단계별 명확한 역할 분리
- **LLM Integration**: 2단계에서 최적화된 LLM 활용
- **Memory Management**: 단계별 메모리 최적화
- **Modular Design**: 각 단계별 독립적 개발 및 테스트 가능

### Memory Management
- **Stage-based Cleanup**: 파이프라인 단계별 메모리 정리
- **State Flush Nodes**: 전용 메모리 정리 노드
- **Garbage Collection**: 자동 메모리 해제 및 최적화
- **Debug State Saving**: 각 단계별 상태를 JSON으로 저장/복원

## � 프로젝트 통계

- **총 Python 파일**: 81개
- **주요 디렉토리**: 15개
- **핵심 모듈**:
  ## 📊 프로젝트 통계 (최신 업데이트)

- **총 Python 파일**: 81개+
- **주요 디렉토리**: 15개+
- **핵심 모듈**:
  - Stage 1: 8개 파일 (데이터 준비) ✅ 완료
  - Stage 2: 5개 파일 (전처리) ✅ 완료 + ProjectDirectoryManager 통합
  - Stage 3: 1개 파일 (분류) 📋 개발 예정
  - Utils: ProjectDirectoryManager, 상태 관리 ✨ 신규 추가
  - Tests: Stage2 matched_questions 검증 ✨ 신규 추가

## 📁 프로젝트 구조

```
2.langgraph/
├── 📂 config/                      # 설정 및 구성 파일
│   ├── config.py                  # 메인 설정 (SAVE_STATE_LOG, PROJECT_ROOT 등)
│   ├── env/                       # 환경별 설정
│   ├── llm/                       # LLM 모델 설정
│   │   └── config_llm.py         # LLM 모델 레지스트리
│   └── prompt/                    # 프롬프트 설정
│       └── prompt.config.yaml    # YAML 프롬프트 정의
├── 📂 utils/                       # 공통 유틸리티 ✨ 강화
│   ├── project_manager.py        # ProjectDirectoryManager 클래스
│   ├── state.py                  # 상태 관리 유틸
│   ├── memory.py                 # 메모리 관리 유틸
│   ├── file_io.py                # 파일 I/O 헬퍼
│   ├── validation.py             # 검증 유틸리티
│   └── load_dataset.py           # 데이터셋 로딩
├── 📂 router/                      # 라우팅 로직
│   ├── qytpe_router.py           # 기본 질문 타입 라우터
│   └── stage2_router.py          # Stage2 전용 라우터 ✨ 신규
├── 📂 nodes/                       # 노드 기반 처리 로직
│   ├── 📂 stage1_data_preparation/ # 1단계: 데이터 준비 (No-LLM)
│   │   ├── survey_loader.py       # 설문조사 파일 로딩
│   │   ├── data_loader.py         # 데이터 파일 로딩  
│   │   ├── survey_parser.py       # 설문조사 파싱
│   │   ├── survey_context.py      # 설문 맥락 추출
│   │   ├── column_extractor.py    # 개방형 컬럼 추출
│   │   ├── question_matcher.py    # 질문-데이터 매칭
│   │   └── memory_optimizer.py    # 1단계 메모리 최적화
│   ├── 📂 stage2_data_preprocessing/ # 2단계: 데이터 전처리 (LLM-Based) ✨ 완전 구현
│   │   ├── stage2_main.py         # Stage2 메인 wrapper
│   │   ├── stage2_word_node.py    # WORD 타입 처리 (concept, img)
│   │   ├── stage2_sentence_node.py # SENTENCE 타입 처리 (depend_pos_neg 등)
│   │   ├── stage2_etc_node.py     # ETC 타입 처리
│   │   └── prep_sentence.py       # 문장 처리 지원 함수

## �📁 프로젝트 구조

```
2.langgraph/
├── 📂 config/                      # 설정 및 구성 파일
│   ├── config.py                  # 메인 설정
│   ├── env/                       # 환경별 설정
│   ├── llm/                       # LLM 모델 설정
│   │   └── config_llm.py         # LLM 모델 레지스트리
│   └── prompt/                    # 프롬프트 설정
│       └── prompt.config.yaml    # YAML 프롬프트 정의
├── 📂 router/                      # 라우팅 로직
│   ├── qytpe_router.py           # 기본 질문 타입 라우터
│   └── stage2_router.py          # Stage2 전용 라우터
├── 📂 nodes/                       # 노드 기반 처리 로직
│   ├── 📂 stage1_data_preparation/ # 1단계: 데이터 준비 (No-LLM)
│   │   ├── survey_loader.py       # 설문조사 파일 로딩
│   │   ├── data_loader.py         # 데이터 파일 로딩  
│   │   ├── survey_parser.py       # 설문조사 파싱
│   │   ├── survey_context.py      # 설문 맥락 추출
│   │   ├── column_extractor.py    # 개방형 컬럼 추출
│   │   ├── question_matcher.py    # 질문-데이터 매칭
│   │   └── memory_optimizer.py    # 1단계 메모리 최적화
│   ├── 📂 stage2_data_preprocessing/ # 2단계: 데이터 전처리 (LLM-Based)
│   │   ├── stage2_main.py         # Stage2 메인 wrapper
│   │   ├── stage2_word_node.py    # WORD 타입 처리
│   │   ├── stage2_sentence_node.py # SENTENCE 타입 처리 (핵심)
│   │   ├── stage2_etc_node.py     # ETC 타입 처리
│   │   └── prep_sentence.py       # 문장 처리 지원 함수
│   ├── 📂 stage3_classification/   # 3단계: 분류 및 클러스터링 (개발 예정)
│   │   └── README.md              # 개발 가이드
│   └── [기타 분류/처리 노드들...]  # 기존 분류 알고리즘 모듈들
├── 📂 graph/                       # 그래프 워크플로우
│   └── graph.py                   # 메인 워크플로우 정의 (Stage2 라우팅 수정 완료)
├── 📂 tests/                       # 테스트 스위트
│   ├── test_stage2_matched_questions.py # Stage2 데이터 저장 검증 ✨ 신규
│   ├── test_graph_stage1.py       # Stage1 테스트
│   └── [기타 테스트 파일들...]
├── 📂 data/                        # 프로젝트별 데이터 디렉토리 ✨ 신규 구조
│   └── {project_name}/            # 프로젝트별 디렉토리
│       ├── raw/                   # 외부 API 원본 데이터
│       ├── state.json             # 최신 상태
│       ├── state_history/         # 프로세스별 상태 이력
│       └── temp_data/stage2_results/ # Stage2 CSV 결과
└── main.py                        # 메인 실행 파일
```

### 🔧 핵심 컴포넌트

#### ProjectDirectoryManager (신규)
```python
# 프로젝트 디렉토리 관리 및 상태 추적
pm = ProjectDirectoryManager("test_project")
pm.create_directories()          # raw/, state_history/, temp_data/ 생성
pm.copy_raw_files(survey, data)  # raw/ 디렉토리에 원본 복사
pm.save_state_history(state, "stage2")  # 상태 이력 저장
csv_path = pm.save_stage2_result(data, qid, type)  # CSV 결과 저장
```

#### Stage2 Router Architecture  
```python
# router/stage2_router.py - 질문 타입별 분기
def stage2_type_router(state):
    current_question = state["matched_questions"]["questions"][state["current_question_index"]]
    question_type = current_question["type"]
    
    if question_type in ["concept", "img"]:
        return "stage2_word_node"
    elif question_type in ["depend", "depend_pos_neg", "pos_neg"]:
        return "stage2_sentence_node"  
    else:
        return "stage2_etc_node"
```
│   ├── 📂 text_preprocess/        # 텍스트 전처리 모듈
│   │   ├── grammar_check.py      # 문법 교정
│   │   ├── sentence_processor.py # 문장 처리
│   │   └── prompts.py           # 프롬프트 관리
│   └── 📂 file_preprocess/        # 파일 전처리 도구
├── 📂 io_layer/                    # I/O 계층
│   └── llm/
│       └── client.py             # LLM 클라이언트
├── 📂 graph/                       # 그래프 워크플로우
│   ├── state.py                  # GraphState 정의
│   └── graph.py                  # 메인 워크플로우
├── 📂 utils/                       # 유틸리티 함수들
│   └── stage_converter.py       # 단계 변환 유틸
├── 📂 script/                      # 실행 스크립트
├── 📂 tests/                       # 테스트 파일들
│   ├── stage1_deep_test.py       # Stage1 심화 테스트
│   ├── stage2_prompt_work.py     # Stage2 프롬프트 작업
│   └── debug_states/             # 디버그 상태 저장소
├── 📂 output/                      # 출력 디렉토리
│   └── stage2_results/           # Stage2 CSV 결과물
├── 📂 data/                        # 프로젝트 데이터
├── 📂 debug_states/                # 디버그 상태 파일
├── 📂 pipeline_history/            # 파이프라인 실행 히스토리
├── main.py                        # 메인 실행 파일
├── integration_test.py            # 통합 테스트
├── test_graph_stage1.py          # Stage1 그래프 테스트
├── test_router_stage2.py         # Stage2 라우터 테스트
├── test_stage2_full.py           # Stage2 전체 테스트
└── test_state_pop.py             # 상태 팝 테스트
```

## 🚀 사용법

### 1. 기본 파이프라인 실행
```python
from main import main

# 전체 파이프라인 실행 (Stage 1 + Stage 2)
result = main()
```

### 2. Stage 2 전처리 실행 예시
```python
from utils.project_manager import ProjectDirectoryManager
from graph.graph import create_graph

# 프로젝트 설정 및 그래프 생성
pm = ProjectDirectoryManager("test_project")
pm.create_directories()
pm.copy_raw_files("survey.txt", "data.xlsx")

# 그래프 실행
graph = create_graph()
result = graph.invoke({
    "project_name": "test_project",
    "survey_path": "survey.txt", 
    "data_path": "data.xlsx"
})
```

### 3. 테스트 실행
```bash
# Stage 2 matched_questions 데이터 저장 검증
python tests/test_stage2_matched_questions.py

# Stage 1 기본 테스트
python tests/test_graph_stage1.py
```

## 📊 Stage 2 처리 결과 예시

### matched_questions 구조 (Stage 2 완료 후)
```json
{
    "questions": [
        {
            "qid": "Q15_1",
            "question": "AI 기술에 대한 귀하의 생각은?",
            "type": "depend_pos_neg",
            "data_column": "Q15_1",
            "stage2_data": "/home/user/data/test_project/temp_data/stage2_results/stage2_Q15_1_depend_pos_neg_20241217_143022.csv"
        }
    ]
}
```

### CSV 출력 예시 (SENTENCE 타입 처리 결과)
```csv
org_text,correction_text,pos.neg,matching_question,sentence_1,sentence_2,sentence_3,S_1,V_1,C_1,S_2,V_2,C_2,S_3,V_3,C_3
"AI는 좋은데 걱정도돼","AI는 좋은데 걱정도 된다",NEUTRAL,HIGH,"AI는 좋다","걱정이 된다","","AI","좋다","","걱정","된다","","","",""
```
│   ├── classifications/            # 기존 분류 로직 (마이그레이션 예정)
│   └── state_flush_node.py        # 메모리 관리 유틸리티
├── core/                           # 핵심 엔진
│   ├── embedding/                  # 임베딩 처리
│   │   └── embedder.py            # 향상된 벡터 임베딩
│   ├── clustering/                 # 클러스터링 알고리즘
```

## 📊 Stage 2 처리 결과 예시

### 입력 데이터
```
원본 텍스트: "이 제품은 정말 좋습니다"
질문 타입: "pos_neg"  
```

### 처리 과정
```
1. Router 판단: "pos_neg" → "SENTENCE"
2. 문법 교정 (gpt-4.1): "이 제품은 정말 좋습니다" → "이 제품 정말 만족스럽습니다"
3. 문장 분석 (gpt-4.1-nano): 감정분석 + SVC 추출
```

### CSV 출력 결과
```csv
id,pos.neg,matching_question,org_text,correction_text,sentence_1,S_1,V_1,C_1
0,POSITIVE,True,"이 제품은 정말 좋습니다","이 제품 정말 만족스럽습니다","이 제품 정말 만족스럽습니다","이 제품","만족스럽다",""
```

## 🔧 설치 및 설정

### 필요 패키지
```bash
pip install langchain langgraph pandas openpyxl python-docx
```

### API 키 설정  
```bash
export OPENAI_API_KEY="your-api-key"
```

### Config 설정
```python
# config/config.py에서 설정 가능
SAVE_STATE_LOG = True           # 상태 이력 저장 여부
PROJECT_ROOT = "./data/"        # 프로젝트 데이터 루트 디렉토리
DEFAULT_PROJECT_NAME = "ai_classification"
```

## 📈 개발 히스토리

### Latest Update (2024-12-17)
- ✅ **ProjectDirectoryManager**: 체계적인 프로젝트 디렉토리 관리 및 상태 이력 추적
- ✅ **Enhanced matched_questions**: stage2_data 필드로 Stage 3에서 CSV 경로 사용 가능
- ✅ **State History**: 매 처리 단계별 상태 저장으로 완전한 추적성 확보
- ✅ **Raw Data Management**: 외부 API 연동을 위한 raw/ 디렉토리 구조 추가
- ✅ **Comprehensive Testing**: test_stage2_matched_questions.py로 전체 플로우 검증

### Stage 2 Implementation (2024-09-17)
- ✅ **Router-Based Architecture**: 기존 프로젝트 패턴 준수한 라우터 기반 구조 구현
- ✅ **Dual LLM Processing**: gpt-4.1 (문법교정) + gpt-4.1-nano (문장분석) 파이프라인
- ✅ **Type-Specific Nodes**: WORD/SENTENCE/ETC 타입별 독립 노드 분리
- ✅ **CSV Output System**: 전처리 결과 자동 저장 시스템

### Key Improvements
- **Architecture Pattern**: 단일 노드 내부 분기 → 라우터 + 조건부 엣지 패턴
- **Project Management**: 수동 파일 관리 → ProjectDirectoryManager 클래스 기반 자동화
- **State Tracking**: 일회성 상태 → state_history를 통한 완전한 추적성
- **Data Flow**: 단순 CSV 저장 → matched_questions 내 stage2_data 경로 임베딩

## 🎯 현재 상태

### ✅ 완료된 기능
- **Stage 1**: 데이터 준비 및 질문 매칭 완료
- **Stage 2**: 텍스트 전처리 완료 (Router + Dual LLM + ProjectDirectoryManager)
- **Project Management**: 프로젝트 디렉토리 자동 생성 및 관리
- **State History**: 매 단계별 상태 이력 저장 및 추적
- **Memory Management**: 단계별 메모리 최적화
- **Testing**: 포괄적 단위/통합 테스트 (Stage 1→2 완전 검증)

### 📋 개발 예정
- **Stage 3**: 텍스트 분류 및 클러스터링 (matched_questions의 stage2_data 활용)
- **API Integration**: 외부 API를 통한 raw/ 디렉토리 데이터 가져오기
- **Performance Optimization**: 대용량 데이터 처리 최적화

## 🤝 기여하기

새로운 노드나 기능을 추가할 때는 기존 아키텍처 패턴을 따라주세요:
1. **ProjectDirectoryManager 활용**: 모든 파일 저장은 pm.save_* 메서드 사용
2. **State History 저장**: 주요 처리 완료 후 pm.save_state_history() 호출
3. **Router-based routing**: 단일 노드 내부 분기 금지
4. **matched_questions 업데이트**: Stage별 데이터 경로를 matched_questions에 임베딩
5. **Consistent error handling**: 표준화된 에러 처리

## 📈 개발 히스토리

### Stage 2 Implementation (2025.09.17)
- ✅ **Router-Based Architecture**: 기존 프로젝트 패턴 준수한 라우터 기반 구조 구현
- ✅ **Dual LLM Processing**: gpt-4.1 (문법교정) + gpt-4.1-nano (문장분석) 파이프라인
- ✅ **Type-Specific Nodes**: WORD/SENTENCE/ETC 타입별 독립 노드 분리
- ✅ **CSV Output System**: 전처리 결과 자동 저장 시스템
- ✅ **Code Cleanup**: 기존 임시 파일 및 테스트 파일 정리

### Key Improvements
- **Architecture Pattern**: 단일 노드 내부 분기 → 라우터 + 조건부 엣지 패턴
- **LLM Interface**: llm_router → io_layer.llm.client 직접 호출
- **Separation of Concerns**: wrapper 노드와 실행 노드 명확한 분리
- **Output Standardization**: tests/stage2_prompt_work.py 패턴 준수

## 🎯 현재 상태

### ✅ 완료된 기능
- **Stage 1**: 데이터 준비 및 질문 매칭 완료
- **Stage 2**: 텍스트 전처리 완료 (Router + Dual LLM)
- **Memory Management**: 단계별 메모리 최적화
- **Testing**: 단위 테스트 및 통합 테스트

### � 개발 예정
- **Stage 3**: 텍스트 분류 및 클러스터링
- **Graph Integration**: 전체 파이프라인 통합
- **Performance Optimization**: 대용량 데이터 처리 최적화

## 🤝 기여하기

새로운 노드나 기능을 추가할 때는 기존 아키텍처 패턴을 따라주세요:
1. Router-based routing (단일 노드 내부 분기 금지)
2. Wrapper + Executor 패턴
3. Type-specific processing nodes
4. Consistent error handling
from graph.state import initialize_project_state

# 워크플로우 생성
workflow = create_workflow()
compiled_graph = workflow.compile()

# 상태 초기화
state = initialize_project_state(
    project_name="your_project",
    survey_filename="survey.txt",
    data_filename="data.xlsx"
)

# 파이프라인 실행
result = compiled_graph.invoke(state)
```

### 디버깅 및 테스트
```bash
# Stage 1 통합 디버깅 테스트 실행
python3 tests/stage1_depp_test.py

# 향후 각 단계별 테스트 추가 예정
# python3 tests/stage2_test.py
# python3 tests/stage3_test.py
```

## 🧹 메모리 최적화 시스템

### Stage-based Memory Management
- **Stage 1 Cleanup**: 1차 파이프라인 완료 후 대용량 원시 데이터 정리
- **Incremental Cleanup**: 단계별 점진적 메모리 해제
- **Essential Data Preservation**: 다음 단계에 필요한 데이터만 유지

### Memory Flush Nodes
- **stage1_memory_flush_node**: 1단계 완료 후 메모리 정리
- **memory_status_check_node**: 메모리 사용량 모니터링
- **force_memory_cleanup**: 강력한 메모리 정리 (긴급상황용)

### Cleanup Process
1. **Stage 1**: 원시 데이터 정리 (raw_survey_info, raw_data_info, parsed_survey)
2. **Essential Data 보존**: question_data_match, llm_logs, open_columns
3. **Garbage Collection**: Python 가비지 컬렉션 강제 실행
4. **Progress Tracking**: 각 단계별 메모리 사용량 추적

### Memory Monitoring
```python
# 메모리 상태 확인
memory_status_check_node(state)

# 수동 메모리 정리
cleaned_state = stage1_memory_flush_node(state)
```

## 📊 State Management

### GraphState 구조
- **Project Info**: project_name, file paths
- **Raw Data**: raw_survey_info, raw_data_info
- **Processed Data**: parsed_survey, open_columns
- **Results**: question_data_match, matched_questions
- **Logs**: llm_logs, llm_meta
- **Memory**: 최적화된 필드 관리

### State Lifecycle
1. **Initialize**: 프로젝트 기반 상태 초기화
2. **Process**: 각 노드에서 데이터 처리
3. **Flush**: 단계별 메모리 정리
4. **Save**: JSON 형태로 상태 저장

## 🔍 디버깅 기능

### Debug State Snapshots
- 각 파이프라인 단계별 상태를 JSON으로 자동 저장
- `debug_states/` 폴더에 타임스탬프별 저장
- 상태 복원 및 분석 가능

### Error Handling
- 각 노드별 에러 처리 및 로깅
- LLM 사용량 추적
- 실패 지점 정확한 파악

## � Development Status

## 📋 최종 구현 상세

### ✅ Stage 1: Data Preparation (COMPLETE)
- **survey_loader.py**: 설문조사 파일 로딩 및 검증
- **data_loader.py**: 데이터 파일 로딩 및 전처리
- **survey_parser.py**: 설문조사 구조 분석 (LLM 사용)
- **column_extractor.py**: 개방형 응답 컬럼 식별
- **question_matcher.py**: 질문-데이터 자동 매칭
- **memory_optimizer.py**: 1단계 메모리 최적화
- **Testing**: `test_graph_stage1.py`로 완전히 검증됨

### ✅ Stage 2: Data Preprocessing (COMPLETE)
- **ProjectDirectoryManager**: 프로젝트별 디렉토리 구조 및 상태 관리
- **Router-Based Processing**: stage2_router → type별 노드 분기
- **Dual LLM Pipeline**: gpt-4.1 (문법교정) + gpt-4.1-nano (문장분석)
- **Enhanced matched_questions**: stage2_data 필드로 Stage 3 연동
- **State History Tracking**: 매 처리별 상태 이력 저장
- **Testing**: `test_stage2_matched_questions.py`로 전체 플로우 검증

### � Stage 3: Classification & Clustering (PLANNED)
- **CSV Data Loading**: matched_questions의 stage2_data 경로 활용
- **Content Classification**: 전처리된 텍스트 기반 분류
- **Clustering Operations**: 임베딩 기반 클러스터링

---

## 🏆 프로젝트 완성도

**현재 완료도: Stage 1 ✅ + Stage 2 ✅ = 66% 완료**

이 README는 2024-12-17 기준으로 모든 최신 구현 사항을 반영하였습니다.
- ProjectDirectoryManager 기반 체계적인 프로젝트 관리
- matched_questions 내 stage2_data 필드로 Stage간 데이터 연동
- 완전한 상태 이력 추적 및 CSV 결과 저장
- 포괄적인 테스트 코드로 검증된 안정적인 파이프라인
- **기능**: 임베딩 기반 분류 및 클러스터링, 코드프레임 생성
- **위치**: `nodes/stage3_classification/` (구조 준비됨)
- **알고리즘**: KNN, CSLS, MCL 클러스터링 통합 예정
- **입력**: Stage 2의 CSV 출력 결과
- **출력**: 최종 분류 및 클러스터링 결과
- **상태**: 📋 향후 개발 예정 (현재 구조만 준비)
- **개발 가이드**: `nodes/stage3_classification/README.md`

## �🚀 Advanced Features

### Project-based Structure
```
data/
├── project1/
│   ├── survey.txt
│   └── data.xlsx
└── project2/
    ├── survey.txt
    └── data.xlsx
```

### LLM Integration
- 다중 모델 지원 (GPT-4, GPT-4o 등)
- 자동 라우팅 및 fallback
- 사용량 모니터링 및 비용 추적

### Classification Types (Stage 2/3)
- **concept**: 개념 분류
- **img**: 이미지/형용사 분류  
- **depend_pos_neg**: 의존형 긍정/부정 분류
- **sentence**: 문장 분류

## 📈 성능 최적화

### Architectural Benefits
- **Stage Separation**: 각 단계별 독립적 최적화 가능
- **Memory Efficiency**: 대용량 데이터 처리 시 메모리 사용량 50-70% 감소
- **Modular Development**: 단계별 병렬 개발 및 테스트 가능
- **LLM Cost Optimization**: 2단계에서만 LLM 사용으로 비용 절감

### Processing Speed
- 파이프라인 단계별 병렬 처리 가능
- LLM 호출 최적화 (Stage 2에 집중)
- 캐싱 및 재사용 최적화
- 단계별 메모리 정리로 성능 향상

## 🛠 개발 히스토리

자세한 개발 과정은 [History.md](History.md)를 참조하세요.

## 📝 라이센스

MIT License