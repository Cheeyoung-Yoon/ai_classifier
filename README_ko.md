# AI 텍스트 분류 파이프라인

설문 자료를 읽고 개방형 질문을 응답 데이터와 매칭한 뒤, Stage 2 전처리(문장·단어 처리, 임베딩, CSV 산출)를 수행하는 LangGraph 기반 워크플로우입니다. 대규모 한글 설문 데이터 처리를 위해 설계되었습니다.

---

## 주요 디렉터리

```
2.langgraph/
├── graph/
│   ├── graph.py                # 파이프라인 정의
│   └── state.py                # 상태 스키마 및 초기화
├── nodes/
│   ├── stage1_data_preparation/
│   │   ├── survey_loader.py
│   │   ├── data_loader.py
│   │   ├── survey_parser.py
│   │   ├── survey_context.py
│   │   ├── column_extractor.py
│   │   └── question_matcher.py
│   ├── stage2_data_preprocessing/
│   │   ├── stage2_main.py
│   │   ├── stage2_sentence_node.py
│   │   ├── stage2_word_node.py
│   │   ├── stage2_etc_node.py
│   │   └── prep_sentence.py
│   ├── stage2_next_question.py
│   └── shared/stage_tracker.py
├── router/stage2_router.py
├── io_layer/
│   ├── llm/client.py
│   └── embedding/embedding.py
├── config/prompt/prompt.config.yaml
└── tests/stage1_and_stage2_full_test.py
```

보조 실험용 스크립트(`0.refine/`, `EDA/` 등)는 루트에 함께 있지만 메인 파이프라인과는 분리되어 있습니다.

---

## 빠른 실행

```bash
cd 2.langgraph
python tests/stage1_and_stage2_full_test.py
```

필수 조건:

- `OPENAI_API_KEY` 환경 변수 설정 (`io_layer/llm/client.py` 참고).
- OpenAI 모델(`gpt-4.1`, `gpt-4.1-nano` 등) 접근 가능.
- `sentence_transformers` 및 `jhgan/ko-sroberta-multitask` 모델 설치 (최초 실행 시 자동 다운로드).

테스트는 `test` 프로젝트 샘플 데이터를 사용하여 Stage 1~2 전체 흐름을 실행하고, 결과 CSV를 `data/test/temp_data/stage2_results/` 아래에 생성합니다.

---

## 파이프라인 흐름

### Stage 1 – 데이터 준비

1. **파이프라인 초기화** (`graph/graph.py` → `pipeline_initialization_node`)
   - `utils/project_manager.py`로 프로젝트별 디렉터리(`data/<project>/raw`, `temp_data` 등) 생성.
   - `stage_tracker`가 스테이지 히스토리 및 비용 로깅 설정.

2. **설문/데이터 로드**
   - `survey_loader.py`: `FileLoader`를 사용해 원본 설문 로드.
   - `data_loader.py`: 엑셀을 CSV로 변환하고 경로를 `raw_dataframe_path`에 저장.

3. **설문 파싱**
   - `survey_parser.py`: `tools.llm_router`와 `survey_parser_4.1` 프롬프트로 개방형 문항 추출 및 타입 분류.

4. **설문 맥락 요약**
   - `survey_context.py`: `survey_context_summarizer` 프롬프트로 설문 목적·배경 요약.

5. **오픈 컬럼 탐지**
   - `column_extractor.py`: 메타정보 또는 `DataHelper`로 텍스트 컬럼 분석.

6. **질문-컬럼 매칭**
   - `question_matcher.py`: `question_data_matcher` 프롬프트 호출, 실패 시 폴백 로직 적용.

7. **데이터 통합 & 메모리 정리**
   - `nodes/survey_data_integrate`: 결과를 `matched_questions`로 정규화.
   - `stage1_memory_flush_node`, `memory_status_check_node`: 메모리 정리 및 상태 출력.

각 단계 종료 시 `nodes/shared/stage_tracker.py`가 상태 저장 및 LLM 비용, 실행 시간 등을 기록합니다.

### Stage 2 – 데이터 전처리

Stage 1에서 매칭된 모든 질문을 순회하며 타입별 전처리를 수행합니다.

1. **초기화** (`stage2_main.py`)
   - 첫 질문을 `current_question_id`, `current_question_type`로 설정하고 총 문항 수를 기록.

2. **라우팅** (`router/stage2_router.py`)
   - `QTYPE_MAP`에 따라 `WORD`, `SENTENCE`, `ETC` 브랜치로 분기. `stage2_processing_complete`가 `True`이면 종료.

3. **SENTENCE 처리** (`stage2_sentence_node.py`)
   - `prompt.config.yaml`에서 문법/분석 프롬프트(`sentence_grammar_check`, `sentence_<type>_split`) 로드, 미존재 시 기본 템플릿 사용.
   - LLM 2종 사용: `gpt-4.1`(문법 교정), `gpt-4.1-nano`(감성·문장 분해·키워드 추출).
   - 의존형 문항의 경우 `prep_sentence.extract_question_choices`로 선택지 설명 생성.
   - `ThreadPoolExecutor`로 응답별 병렬 처리 및 비용 누적.
   - `io_layer/embedding/embedding.py`의 `VectorEmbedding`으로 원문, 교정문, 세부 문장들의 임베딩 생성.
   - 결과 CSV(`project_manager.get_stage2_csv_path`)에 감성, 매칭 여부, S/V/C 키워드, 임베딩 벡터 저장.
   - `matched_questions[question_id]['stage2_data']` 갱신 및 설정에 따라 상태 로그 저장.

4. **WORD 처리** (`stage2_word_node.py`)
   - 해당 질문 컬럼 텍스트를 추출, 임베딩 생성 후 CSV로 저장.

5. **ETC 처리** (`stage2_etc_node.py`)
   - 별도 변환 없이 처리 완료 플래그만 설정 (향후 확장 용도).

6. **다음 질문 이동** (`stage2_next_question.py`)
   - 인덱스를 증가시키며 반복, 모든 문항 처리 시 `stage2_processing_complete = True`.

---

## 설정

- **프롬프트** (`config/prompt/prompt.config.yaml`): Stage 1/2에서 사용하는 LLM 프롬프트 정의. 새 타입 지원 시 브랜치 추가.
- **LLM 구성** (`config/llm/config_llm.py`): 모델 이름, 비용 단가, 호출 파라미터.
- **전역 설정** (`config/config.py`): `SAVE_STATE_LOG`, 베이스 경로, API 키 등.

프롬프트만 수정하여 동작을 쉽게 조정할 수 있습니다.

---

## 산출물

- **Stage 2 CSV**: `data/<project>/temp_data/stage2_results/stage2_<question_id>_<type>_<timestamp>.csv`
- **상태 & 히스토리**:
  - 최신 상태: `data/<project>/state.json`
  - 전체 히스토리: `data/<project>/state_history/*state.json`
- **로그**: 각 노드 진행 상황, 비용, 파일 경로를 콘솔에 출력.

CSV에는 응답 ID, 원문/교정문, 임베딩, 세부 문장, 감성, S/V/C 키워드 등이 포함됩니다.

---

## 테스트 & 검증

- `tests/stage1_and_stage2_full_test.py`: Stage 1~2 엔드투엔드 테스트 및 CSV 생성 확인.
- 외부 의존성(LLM, 임베딩)을 피하고 싶다면 `LLMClient`, `VectorEmbedding`을 목 객체로 교체하여 테스트하세요.

---

## 문제 해결

- **프롬프트 누락**: `stage2_sentence_node`에서 경고 로그 출력. `prompt.config.yaml`에 브랜치 추가 필요.
- **CSV 미생성**: `matched_questions`에 문항/타입이 들어있는지, 프로젝트 디렉터리가 생성되었는지 확인.
- **임베딩 오류**: `sentence_transformers` 모델 다운로드 여부와 환경(GPU/CPU) 지원 확인.
- **비용 추적**: 로그에서 문항별 비용을 확인하고, `total_llm_cost_usd`가 누적됩니다.

---

## 향후 계획

- Stage 3(클러스터링, 토픽 분석 등) 모델링 로직 추가.
- 새로운 질문 타입(`etc_pos_neg` 등) 대응 라우터/프롬프트 확장.
- LLM 응답을 고정값으로 대체한 회귀 테스트 작성으로 회귀 검증 강화.
