
"""
Sentence Preprocessing Engine for Stage 2/3 Pipeline

This module handles advanced sentence preprocessing with different strategies
based on question types. Integrates with the 3-stage architecture.

Processing Workflows:
===================

DEPENDENT TYPE (depend):
1. 데이터는 문장 형태, related_Question에서의 값 기반하여 해석된 값을 추가필요
2. 각 문장은 S,V,C로 분해 하여 키워드 추출 (Stage 2)
3. 각 문장은 추출된 S,V,C 기반, 원자적 의미를 가지도록 최대 3개의 문장을 생성 (Stage 2)
4. 각 문장을 embedding 처리 (Stage 3)
5. embedding 한 결과 KNN -> CSLS -> MCL 기반 mapping 진행 (Stage 3)
6. mapping 한 결과를 key로 묶음 (Stage 3)
7. key 기반하여 유의미한 매핑이 있는지 다시 확인 (Stage 3)
8. LLM 확인하여 동일 의미 또는 오탈자로 인한 group 있는지 확인 (Stage 2)
9. LLM 활용하여 S,V 기반 요약된 key 에 대한 요약 문장 생성 (Stage 2)
10. 매핑 테이블 생성 (Stage 3)

SENTENCE TYPE (sentence):
1. 각 문장은 S,V,C로 분해 하여 키워드 추출 (Stage 2)
2. 각 문장은 추출된 S,V,C 기반, 원자적 의미를 가지도록 최대 3개의 문장을 생성 (Stage 2)
3. 각 문장을 embedding 처리 (Stage 3)
4. embedding 한 결과 KNN -> CSLS -> MCL 기반 mapping 진행 (Stage 3)
5. mapping 한 결과를 key로 묶음 (Stage 3)
6. key 기반하여 유의미한 매핑이 있는지 다시 확인 (Stage 3)
7. LLM 확인하여 동일 의미 또는 오탈자로 인한 group 있는지 확인 (Stage 2)
8. LLM 활용하여 S,V 기반 요약된 key 에 대한 요약 문장 생성 (Stage 2)
9. 매핑 테이블 생성 (Stage 3)
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import json


class SentencePreprocessor:
    """
    Advanced sentence preprocessor for different question types.
    
    This class serves as a foundation for Stage 2 (LLM-based preprocessing)
    and Stage 3 (embedding & clustering) operations.
    
    Based on step2_test.py workflow patterns.
    """
    
    def __init__(self, llm_client=None):
        # TODO: Initialize LLM router, embedding model, clustering algorithms
        # when moving to actual implementation
        self.llm_client = llm_client
        pass   
    
    def get_column_locations(self, qid: str, question_data: Dict, state: Dict, 
                           df_path: str = None, with_depend: bool = False) -> Any:
        """
        Extract column data for processing based on step2_test.py pattern.
        
        Args:
            qid: Question ID
            question_data: Question metadata
            state: Current state
            df_path: Path to dataframe
            with_depend: Whether to include dependent column
            
        Returns:
            DataFrame partition or tuple(depend_col, partition)
        """
        if not df_path:
            df_path = state.get('dataframe', './data/suv/raw_data.csv')
        
        df = pd.read_csv(df_path)
        matched_cols = question_data['matched_columns']
        
        if not matched_cols:
            return pd.DataFrame()  # Empty for questions with no matched columns
        
        df_col_loc_list = [df.columns.get_loc(col) for col in matched_cols]
        partition = df[df.columns[df_col_loc_list]]
        
        if with_depend:
            depend_col_idx = min(df_col_loc_list) - 1
            depend_col = df[df.columns[depend_col_idx:depend_col_idx+1]]
            
            # Check if depend column has data
            if depend_col.isna().sum().sum() == len(depend_col):
                depend_col_idx = min(df_col_loc_list) - 2
                depend_col = df[df.columns[depend_col_idx:depend_col_idx+1]]
            
            return depend_col, partition
        
        return partition
    
    def _analyze_question_with_llm(self, question_text: str) -> tuple:
        """
        Analyze question with LLM to extract structure and choices.
        
        Args:
            question_text: Question to analyze
            
        Returns:
            Tuple of (question_summary, answer_choices)
        """
        if not self.llm_client:
            return question_text, {}
        
        system_prompt = """
        Based on the text given, understand the answer labels of the multiple choice question.
        Return the question summary and all choices with explanations in JSON format.
        
        Return in the following JSON format:
        {
            "question": "summary of the question",
            "answers": {
                "1": "explanation for answer 1",
                "2": "explanation for answer 2", 
                ...
            }
        }
        """
        
        try:
            response = self.llm_client.chat(
                system=system_prompt,
                user=f"question: {question_text}"
            )
            result = json.loads(response[0])
            return result['question'], result['answers']
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return question_text, {}
    
    def _process_single_response(self, response_text: str, question_summary: str, 
                                sub_explanation: str, has_sentiment: bool) -> Dict[str, Any]:
        """
        Process single response with LLM for S,V,C extraction and atomic sentences.
        
        Args:
            response_text: Response to process
            question_summary: Summary of question
            sub_explanation: Additional context
            has_sentiment: Whether to include sentiment analysis
            
        Returns:
            Dict with processing results including SVC keywords and atomic sentences
        """
        if not self.llm_client:
            return {
                "matching_question": True,
                "atomic_sentence": response_text.strip(),
                "SVC_keywords": {"S": [], "V": [], "C": []},
                "processed_text": response_text.strip()
            }
        
        if has_sentiment:
            system_prompt = """You are a survey response interpreter with sentiment analysis.
            Analyze the response and extract:
            1. S,V,C keywords (Subject, Verb, Complement)
            2. Generate atomic sentences (max 3) with clear meaning
            3. Determine sentiment (POSITIVE/NEGATIVE)
            
            Return JSON with:
            {
                "matching_question": true/false,
                "pos_neg": "POSITIVE/NEGATIVE",
                "atomic_sentence": ["sentence1", "sentence2", "sentence3"],
                "SVC_keywords": {
                    "S": ["subject keywords"],
                    "V": ["verb keywords"], 
                    "C": ["complement keywords"]
                }
            }"""
        else:
            system_prompt = """You are a survey response interpreter.
            Analyze the response and extract:
            1. S,V,C keywords (Subject, Verb, Complement)
            2. Generate atomic sentences (max 3) with clear meaning
            
            Return JSON with:
            {
                "matching_question": true/false,
                "atomic_sentence": ["sentence1", "sentence2", "sentence3"],
                "SVC_keywords": {
                    "S": ["subject keywords"],
                    "V": ["verb keywords"],
                    "C": ["complement keywords"]
                }
            }"""
        
        user_prompt = f"""
        question: {question_summary}
        sub_explanation: {sub_explanation}
        answer: {response_text}
        """
        
        try:
            response = self.llm_client.chat(system=system_prompt, user=user_prompt)
            result = json.loads(response[0])
            
            # Ensure atomic_sentence is always a list
            if isinstance(result.get('atomic_sentence'), str):
                result['atomic_sentence'] = [result['atomic_sentence']]
            
            return result
        except Exception as e:
            print(f"Response processing failed: {e}")
            return {
                "matching_question": False, 
                "error": f"Processing failed: {str(e)}",
                "atomic_sentence": [response_text.strip()],
                "SVC_keywords": {"S": [], "V": [], "C": []}
            }   
    
    def _dependent_preprocess(self, text: str, related_question: str, 
                             qid: str = None, question_data: Dict = None, 
                             state: Dict = None) -> Dict[str, Any]:
        """
        Preprocess dependent question type text following step2_test.py workflow.
        
        Args:
            text: Input sentence text (or can be extracted from state)
            related_question: Related question for context
            qid: Question ID for column extraction
            question_data: Question metadata
            state: Current state with dataframe info
            
        Returns:
            Dict containing preprocessing results and metadata
        """
        # If state provided, extract actual data from columns
        if state and question_data and qid:
            depend_col, partition = self.get_column_locations(qid, question_data, state, with_depend=True)
            
            # Get question analysis from LLM
            question_text = question_data['question_info']['question_text']
            question_summary, answer_choices = self._analyze_question_with_llm(question_text)
            
            # Process sample responses
            partition["merged"] = partition.fillna("").astype(str).agg(" ".join, axis=1)
            processed_samples = []
            
            # Process first 3 samples as test (following step2_test.py pattern)
            for i in range(min(3, len(partition))):
                sample_result = self._process_single_response(
                    partition.iloc[i]["merged"], 
                    question_summary, 
                    answer_choices.get(str(int(depend_col.iloc[i].values[0])), ""),
                    has_sentiment=False
                )
                processed_samples.append(sample_result)
            
            return {
                "original_text": text,
                "qid": qid,
                "question_type": "dependent",
                "node_type": "SENTENCE",
                "subtype": "depend",
                "question_summary": question_summary,
                "answer_choices": answer_choices,
                "processed_samples": processed_samples,
                "total_samples": len(partition),
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
        else:
            # Simple processing without full workflow
            processed_text = text.strip()
            return {
                "original_text": text,
                "processed_text": processed_text,
                "related_question": related_question,
                "question_type": "dependent",
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
    
    def _pos_neg_preprocess(self, text: str, qid: str = None, 
                           question_data: Dict = None, state: Dict = None) -> Dict[str, Any]:
        """
        Preprocess positive/negative classification text following step2_test.py workflow.
        
        Args:
            text: Input sentence text
            qid: Question ID for column extraction
            question_data: Question metadata  
            state: Current state with dataframe info
            
        Returns:
            Dict containing preprocessing results and metadata
        """
        # If state provided, extract actual data from columns
        if state and question_data and qid:
            partition = self.get_column_locations(qid, question_data, state)
            question_text = question_data['question_info']['question_text']
            
            partition["merged"] = partition.fillna("").astype(str).agg(" ".join, axis=1)
            processed_samples = []
            
            # Process first 3 samples as test
            for i in range(min(3, len(partition))):
                sample_result = self._process_single_response(
                    partition.iloc[i]["merged"], 
                    question_text, 
                    "",
                    has_sentiment=True
                )
                processed_samples.append(sample_result)
            
            return {
                "original_text": text,
                "qid": qid,
                "question_type": "pos_neg",
                "node_type": "SENTENCE",
                "subtype": "pos_neg",
                "processed_samples": processed_samples,
                "total_samples": len(partition),
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
        else:
            # Simple processing without full workflow
            processed_text = text.strip()
            return {
                "original_text": text,
                "processed_text": processed_text,
                "question_type": "pos_neg",
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
    
    def _dependent_pos_neg_preprocess(self, text: str, related_question: str,
                                     qid: str = None, question_data: Dict = None, 
                                     state: Dict = None) -> Dict[str, Any]:
        """
        Preprocess dependent positive/negative classification text following step2_test.py workflow.
        
        Args:
            text: Input sentence text
            related_question: Related question for context
            qid: Question ID for column extraction
            question_data: Question metadata
            state: Current state with dataframe info
            
        Returns:
            Dict containing preprocessing results and metadata
        """
        # If state provided, extract actual data from columns
        if state and question_data and qid:
            depend_col, partition = self.get_column_locations(qid, question_data, state, with_depend=True)
            question_text = question_data['question_info']['question_text']
            
            # Get question analysis
            question_summary, answer_choices = self._analyze_question_with_llm(question_text)
            
            partition["merged"] = partition.fillna("").astype(str).agg(" ".join, axis=1)
            processed_samples = []
            
            # Process first 3 samples as test
            for i in range(min(3, len(partition))):
                sample_result = self._process_single_response(
                    partition.iloc[i]["merged"], 
                    question_summary, 
                    answer_choices.get(str(int(depend_col.iloc[i].values[0])), ""),
                    has_sentiment=True
                )
                processed_samples.append(sample_result)
            
            return {
                "original_text": text,
                "qid": qid,
                "question_type": "dependent_pos_neg",
                "node_type": "SENTENCE",
                "subtype": "depend_pos_neg",
                "question_summary": question_summary,
                "answer_choices": answer_choices,
                "processed_samples": processed_samples,
                "total_samples": len(partition),
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
        else:
            # Simple processing without full workflow
            processed_text = text.strip()
            return {
                "original_text": text,
                "processed_text": processed_text,
                "related_question": related_question,
                "question_type": "dependent_pos_neg",
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
    
    def _sentence_preprocess(self, text: str, qid: str = None, 
                           question_data: Dict = None, state: Dict = None) -> Dict[str, Any]:
        """
        Preprocess general sentence type text following step2_test.py workflow.
        
        Args:
            text: Input sentence text
            qid: Question ID for column extraction
            question_data: Question metadata
            state: Current state with dataframe info
            
        Returns:
            Dict containing preprocessing results and metadata
        """
        # If state provided, extract actual data from columns
        if state and question_data and qid:
            partition = self.get_column_locations(qid, question_data, state)
            question_text = question_data['question_info']['question_text']
            
            partition["merged"] = partition.fillna("").astype(str).agg(" ".join, axis=1)
            processed_samples = []
            
            # Process first 3 samples as test
            for i in range(min(3, len(partition))):
                sample_result = self._process_single_response(
                    partition.iloc[i]["merged"], 
                    question_text, 
                    "",
                    has_sentiment=False
                )
                processed_samples.append(sample_result)
            
            return {
                "original_text": text,
                "qid": qid,
                "question_type": "sentence",
                "node_type": "SENTENCE",
                "subtype": "basic",
                "processed_samples": processed_samples,
                "total_samples": len(partition),
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
        else:
            # Simple processing without full workflow
            processed_text = text.strip()
            return {
                "original_text": text,
                "processed_text": processed_text,
                "question_type": "sentence",
                "stage": "stage2_preprocessing",
                "next_stage": "stage3_embedding_clustering"
            }
    
    def preprocess(self, text: str, q_type: str = None, related_question: str = None,
                   qid: str = None, question_data: Dict = None, state: Dict = None) -> Dict[str, Any]:
        """
        Main preprocessing entry point with question type routing following step2_test.py workflow.
        
        Args:
            text: Input text to preprocess
            q_type: Question type for routing logic
            related_question: Optional related question for dependent types
            qid: Question ID for column extraction
            question_data: Question metadata containing matched_columns and question_info
            state: Current state with dataframe path and other context
            
        Returns:
            Dict containing preprocessing results and metadata
            
        Raises:
            ValueError: If question type is not supported
        """
        if q_type == "depend":
            return self._dependent_preprocess(text, related_question, qid, question_data, state)
        elif q_type == "pos_neg":
            return self._pos_neg_preprocess(text, qid, question_data, state)
        elif q_type == "depend_pos_neg":
            return self._dependent_pos_neg_preprocess(text, related_question, qid, question_data, state)
        elif q_type == "sentence":
            return self._sentence_preprocess(text, qid, question_data, state)
        else:
            raise ValueError(f"Unsupported question type: {q_type}. "
                           f"Supported types: ['depend', 'pos_neg', 'depend_pos_neg', 'sentence']")
    
    def process_question(self, qid: str, state: Dict, llm_client=None) -> Dict[str, Any]:
        """
        Process a complete question following step2_test.py workflow.
        
        This is the main entry point that mirrors the step2_test.py node execution pattern.
        
        Args:
            qid: Question ID to process
            state: State containing matched_questions and dataframe info
            llm_client: Optional LLM client for processing
            
        Returns:
            Dict containing complete processing results
        """
        if llm_client:
            self.llm_client = llm_client
            
        # Get question data from state
        question_data = state.get('matched_questions', {}).get(qid)
        if not question_data:
            return {"error": f"Question {qid} not found in state"}
        
        # Get question type and related info
        question_info = question_data.get('question_info', {})
        qtype = question_info.get('question_type')
        related_question = question_info.get('related_question')
        
        # Route and process based on question type
        return self.preprocess(
            text="",  # Will be extracted from columns
            q_type=qtype,
            related_question=related_question,
            qid=qid,
            question_data=question_data,
            state=state
        )


# Utility functions for future Stage 2/3 implementation

def create_stage2_preprocessing_node(preprocessor: SentencePreprocessor):
    """
    Factory function to create Stage 2 preprocessing node.
    
    This will be used when implementing Stage 2 nodes in:
    nodes/stage2_data_preprocessing/
    """
    def stage2_node(state, text_data, question_type, related_question=None):
        # Future implementation: integrate with LLM router and graph state
        return preprocessor.preprocess(text_data, question_type, related_question)
    
    return stage2_node


def create_stage3_clustering_pipeline():
    """
    Factory function to create Stage 3 embedding & clustering pipeline.
    
    This will be used when implementing Stage 3 nodes in:
    nodes/stage3_classification/
    """
    def stage3_pipeline(preprocessed_data):
        # Future implementation:
        # 1. Embedding generation
        # 2. KNN -> CSLS -> MCL clustering
        # 3. Mapping table generation
        pass
    
    return stage3_pipeline


# Integration hooks for the 3-stage architecture
class Stage2Stage3Integration:
    """
    Integration layer between Stage 2 (preprocessing) and Stage 3 (classification).
    
    This class will handle data flow and state management between stages.
    """
    
    @staticmethod
    def prepare_for_stage3(stage2_output: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Stage 2 output for Stage 3 processing."""
        return {
            "preprocessed_text": stage2_output.get("processed_text"),
            "metadata": {
                "question_type": stage2_output.get("question_type"),
                "related_question": stage2_output.get("related_question"),
                "stage2_complete": True
            }
        }
    
    @staticmethod
    def validate_stage_transition(from_stage: str, to_stage: str, data: Dict[str, Any]) -> bool:
        """Validate that data is ready for stage transition."""
        if from_stage == "stage2_preprocessing" and to_stage == "stage3_embedding_clustering":
            return "processed_text" in data and "question_type" in data
        return False