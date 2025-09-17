"""
Sentence processing module for different question types
"""
import json
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompts import get_branch_by_type
from io_layer.llm.client import LLMClient


def _get_system_prompt_by_type(question_type: str) -> str:
    """Get system prompt based on question type"""
    
    base_rules = """### Rules
1. **Nuance reference**: Context fields are only for reference. Do not include them in the final output.  
2. type-of-error (typo) detection: Identify and correct any typographical errors in the answer including spelling mistakes, grammatical errors, and incorrect word usage.  
3. **Question matching judgment**:  
   - Set `"matching_question": true` unless the answer is clearly irrelevant or meaningless.  
   - Even short or abstract answers like *"trustworthy"*, *"it feels reliable"*, *"good"*, *"satisfying"* should be treated as `true`.  
   - Set `"matching_question": false` only if the answer is irrelevant, empty, meaningless tokens, or pure emotional expression with no semantic relation (e.g., "Wow!", "I'm so excited").  
   - Ignore filler tokens such as `merged`, `nan`, `NaN`, `None`, `NULL`, or `Name: 0, dtype: object` before evaluation.  
4. **Atomic sentence split**: If the answer contains multiple semantic units, split it into at most 3 atomic sentences (Subject–Verb–Complement based).  
5. **S/V/C keyword extraction**: For each atomic sentence, extract core keywords:  
   - S = subject (main entity)  
   - V = verb (main action/state) format in "-다" form
   - C = complement/object (if any)  
   - Only extract essential words, not particles or function words.  
6. **Output format**: Return the result format strictly in the following JSON format (schema stays in Korean):"""

    json_schema = """{
  "matching_question": true/false,
  "pos.neg": "POSITIVE"/"NEGATIVE"/"NEUTRAL",
  "automic_sentence": ["문장1", "문장2", "문장3"],
  "SVC_keywords": {
      "sentence1": {
    "S": [],
    "V": [],
    "C": []},
      "sentence2": {
    "S": [],
    "V": [],
    "C": []},
      "sentence3": {
    "S": [],
    "V": [],
    "C": []
  }
}"""

    if question_type == "depend_pos_neg":
        header = """You are a survey response interpreter.  
You will receive input containing:  
1) survey_context: brief summary of the survey purpose
2) question: the original survey question explanation  
3) sub_explanation: additional clarification of the question's intent  
4) answer: the respondent's free-text answer"""
        
        pos_neg_rule = """7. **Pos/Neg classification**: If the question is about sentiment (positive/negative), classify the answer accordingly with sub_explanation and set `"pos.neg"` to `"POSITIVE"`or `"NEGATIVE"`."""
        
    elif question_type == "pos_neg":
        header = """You are a survey response interpreter.  
You will receive input containing:  
1) survey_context: brief summary of the survey purpose
2) question: the original survey question explanation  
3) answer: the respondent's free-text answer"""
        
        pos_neg_rule = """7. **Pos/Neg classification**: If the question is about sentiment (positive/negative), classify the answer accordingly and set `"pos.neg"` to `"POSITIVE"`or `"NEGATIVE"`."""
        
    elif question_type == "depend":
        header = """You are a survey response interpreter.  
You will receive input containing:  
1) survey_context: brief summary of the survey purpose
2) question: the original survey question explanation  
3) sub_explanation: additional clarification of the question's intent  
4) answer: the respondent's free-text answer"""
        
        pos_neg_rule = """7. pos.neg is just placeholder. just return "NEUTRAL" """
        
    else:  # sentence/etc
        header = """You are a survey response interpreter.  
You will receive input containing:
1) survey_context: brief summary of the survey purpose
2) question: the original survey question explanation
3) answer: the respondent's free-text answer"""
        
        pos_neg_rule = """7. pos.neg is just placeholder. just return "NEUTRAL" """

    return f"{header}\n\n{base_rules}\n{pos_neg_rule}\n{json_schema}"


def _process_single_response(llm_client: LLMClient, question_type: str, variables: Dict[str, Any], row_id: int) -> Tuple[int, Dict]:
    """
    Process a single response with LLM client
    
    Args:
        llm_client: LLM client instance
        question_type: Type of question for prompt selection
        variables: Variables for the prompt template
        row_id: Row identifier for tracking
        
    Returns:
        Tuple of (row_id, parsed_response_dict)
    """
    try:
        system_prompt = _get_system_prompt_by_type(question_type)
        
        # Build user prompt based on question type
        if question_type in ["depend_pos_neg", "depend"]:
            user_prompt = f"""survey_context: {variables.get('survey_context', '')}
question: {variables.get('question_summary', '')}
sub_explanation: {variables.get('sub_explanation', '')}
answer: {variables.get('corrected_answer', '')}"""
        else:
            user_prompt = f"""survey_context: {variables.get('survey_context', '')}
question: {variables.get('question_summary', '')}
answer: {variables.get('corrected_answer', '')}"""
        
        response = llm_client.chat(
            system=system_prompt,
            user=user_prompt
        )
        
        # Parse JSON response
        try:
            data = json.loads(response[0])
        except json.JSONDecodeError:
            # Try to clean up the response if it has markdown formatting
            cleaned_response = response[0].replace('```json', '').replace('```', '').strip()
            data = json.loads(cleaned_response)
            
        return row_id, data
    except Exception as e:
        print(f"Error processing row {row_id}: {e}")
        # Return default structure on error
        default_data = {
            'matching_question': False,
            'pos.neg': 'NEUTRAL',
            'automic_sentence': [],
            'SVC_keywords': {}
        }
        return row_id, default_data


def _create_result_dataframe(results: List[Tuple[int, Dict]], original_texts: List[str], corrected_texts: List[str]) -> pd.DataFrame:
    """
    Create a result DataFrame from processed results
    
    Args:
        results: List of (row_id, data) tuples
        original_texts: List of original text responses
        corrected_texts: List of corrected text responses
        
    Returns:
        DataFrame with all processed results
    """
    result_data = []
    
    for row_id, data in results:
        result_row = {
            'id': row_id,
            'original_text': original_texts[row_id],
            'corrected_text': corrected_texts[row_id],
            'matching_question': data.get('matching_question', False),
            'pos_neg': data.get('pos.neg', 'NEUTRAL'),
            'atomic_sentence_1': data.get('automic_sentence', [None, None, None])[0],
            'atomic_sentence_2': data.get('automic_sentence', [None, None, None])[1],
            'atomic_sentence_3': data.get('automic_sentence', [None, None, None])[2],
        }
        
        # Extract SVC keywords for each sentence
        svc_keywords = data.get('SVC_keywords', {})
        for i in range(1, 4):
            sentence_key = f'sentence{i}'
            svc = svc_keywords.get(sentence_key, {'S': [], 'V': [], 'C': []})
            result_row[f'S{i}'] = svc.get('S', [])
            result_row[f'V{i}'] = svc.get('V', [])
            result_row[f'C{i}'] = svc.get('C', [])
        
        result_data.append(result_row)
    
    return pd.DataFrame(result_data)


def process_depend_pos_neg(llm_client: LLMClient, survey_context: str, question_summary: str, response_dict: Dict, 
                          depend_df: pd.DataFrame, text_df: pd.DataFrame, 
                          corrected_texts: List[str], max_workers: int = 5) -> pd.DataFrame:
    """
    Process depend_pos_neg type questions with multithreading
    
    Args:
        llm_client: LLM client instance  
        survey_context: Survey context summary
        question_summary: Summary of the question
        response_dict: Dictionary mapping depend values to explanations
        depend_df: DataFrame with depend column
        text_df: DataFrame with text data
        corrected_texts: List of grammar-corrected texts
        max_workers: Maximum number of threads to use
        
    Returns:
        DataFrame with processed results
    """
    print(f"Starting depend_pos_neg processing with {len(corrected_texts)} samples")
    result_data = []
    futures_to_row = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, corrected_text in enumerate(corrected_texts):
            depend_value = depend_df.iloc[idx]['depend']
            sub_explanation = response_dict.get(depend_value, "")
            
            variables = {
                'survey_context': survey_context,
                'question_summary': question_summary,
                'sub_explanation': sub_explanation,
                'corrected_answer': corrected_text
            }
            
            future = executor.submit(_process_single_response, llm_client, "depend_pos_neg", variables, idx)
            futures_to_row[future] = idx
        
        # Collect results
        for future in as_completed(futures_to_row):
            row_id, data = future.result()
            
            result_row = {
                'id': row_id,
                'matching_question': data.get('matching_question', False),
                'pos_neg': data.get('pos.neg', 'NEUTRAL'),
                'atomic_sentence_1': data.get('automic_sentence', [None, None, None])[0],
                'atomic_sentence_2': data.get('automic_sentence', [None, None, None])[1],
                'atomic_sentence_3': data.get('automic_sentence', [None, None, None])[2],
            }
            
            # Extract SVC keywords for each sentence
            svc_keywords = data.get('SVC_keywords', {})
            for i in range(1, 4):
                sentence_key = f'sentence{i}'
                svc = svc_keywords.get(sentence_key, {'S': [], 'V': [], 'C': []})
                result_row[f'S{i}'] = svc.get('S', [])
                result_row[f'V{i}'] = svc.get('V', [])
                result_row[f'C{i}'] = svc.get('C', [])
            
            result_data.append(result_row)
    
    return pd.DataFrame(result_data)


def process_pos_neg(llm_client: LLMClient, survey_context: str, question_summary: str, text_df: pd.DataFrame,
                   corrected_texts: List[str], max_workers: int = 5) -> pd.DataFrame:
    """
    Process pos_neg type questions with multithreading
    """
    # Prepare tasks for multithreading
    tasks = []
    for i in range(len(text_df)):
        variables = {
            "survey_context": survey_context,
            "question_summary": question_summary,
            "corrected_answer": corrected_texts[i]
        }
        
        tasks.append((llm_client, "pos_neg", variables, i))
    
    # Process with multithreading
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_process_single_response, *task): task[3] 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            row_id, data = future.result()
            results.append((row_id, data))
    
    # Sort results by row_id to maintain order
    results.sort(key=lambda x: x[0])
    
    # Create result DataFrame
    original_texts = [text_df.iloc[i].values[0].strip() for i in range(len(text_df))]
    return _create_result_dataframe(results, original_texts, corrected_texts)


def process_depend(llm_client: LLMClient, survey_context: str, question_summary: str, response_dict: Dict,
                  depend_df: pd.DataFrame, text_df: pd.DataFrame, 
                  corrected_texts: List[str], max_workers: int = 5) -> pd.DataFrame:
    """
    Process depend type questions with multithreading
    """
    # Prepare tasks for multithreading
    tasks = []
    for i in range(len(depend_df)):
        depend_value = str(int(depend_df.iloc[i, 0]))
        sub_explanation = response_dict.get(depend_value, "")
        
        variables = {
            "survey_context": survey_context,
            "question_summary": question_summary,
            "sub_explanation": sub_explanation,
            "corrected_answer": corrected_texts[i]
        }
        
        tasks.append((llm_client, "depend", variables, i))
    
    # Process with multithreading
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_process_single_response, *task): task[3] 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            row_id, data = future.result()
            results.append((row_id, data))
    
    # Sort results by row_id to maintain order
    results.sort(key=lambda x: x[0])
    
    # Create result DataFrame
    original_texts = [text_df.iloc[i].values[0].strip() for i in range(len(text_df))]
    return _create_result_dataframe(results, original_texts, corrected_texts)


def process_sentence(llm_client: LLMClient, survey_context: str, question_summary: str, text_df: pd.DataFrame,
                    corrected_texts: List[str], max_workers: int = 5) -> pd.DataFrame:
    """
    Process sentence type questions with multithreading
    """
    # Prepare tasks for multithreading
    tasks = []
    for i in range(len(text_df)):
        variables = {
            "survey_context": survey_context,
            "question_summary": question_summary,
            "corrected_answer": corrected_texts[i]
        }
        
        tasks.append((llm_client, "sentence", variables, i))
    
    # Process with multithreading
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_process_single_response, *task): task[3] 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            row_id, data = future.result()
            results.append((row_id, data))
    
    # Sort results by row_id to maintain order
    results.sort(key=lambda x: x[0])
    
    # Create result DataFrame
    original_texts = [text_df.iloc[i].values[0].strip() for i in range(len(text_df))]
    return _create_result_dataframe(results, original_texts, corrected_texts)