"""
Sentence preparation node for stage2 preprocessing
"""
import pandas as pd
import json
import tqdm
from typing import Dict, Any, Optional, List
from io_layer.llm.client import LLMClient
from io_layer.embedding.embedding import VectorEmbedding
from tools.text_preprocess import (
    grammar_check, 
    process_depend_pos_neg, 
    process_pos_neg, 
    process_depend, 
    process_sentence
)


def get_column_locations(state: Dict[str, Any], qid: str, node_type: str = "SENTENCE"):
    """
    Get column locations for a specific question ID
    Based on tests.stage2_prompt_work.py logic
    """
    df_path = state.get('raw_dataframe_path')
    if not df_path:
        raise ValueError("raw_dataframe_path not found in state")
    
    df = pd.read_csv(df_path)
    matched_questions = state.get('matched_questions', {})
    
    if qid not in matched_questions:
        raise ValueError(f"Question ID {qid} not found in matched_questions")
    
    matched_cols = matched_questions[qid]['matched_columns']
    df_col_loc_list = [df.columns.get_loc(col) for col in matched_cols]
    
    # Get the main partition (answer columns)
    partition = df[df.columns[df_col_loc_list]]
    
    if node_type == "SENTENCE" and matched_questions[qid]['question_info']['question_type'] in ["depend", "depend_pos_neg"]:
        # For depend types, get the depend column (usually the column before the first matched column)
        depend_col_idx = min(df_col_loc_list) - 1
        depend_col = df[df.columns[depend_col_idx:depend_col_idx+1]]
        
        # Check if depend column is mostly empty, if so, try the previous column
        if depend_col.isna().sum().sum() == len(depend_col):
            depend_col_idx = min(df_col_loc_list) - 2
            depend_col = df[df.columns[depend_col_idx:depend_col_idx+1]]
        
        return depend_col, partition
    
    return partition


def extract_question_choices(llm_client: LLMClient, question_text: str, unique_numbers: List) -> Dict:
    """
    Extract question choices using LLM client
    """
    try:
        system_prompt = """You are a survey question analyzer. Given a question text and unique response numbers, extract the question summary and map each number to its meaning.

Return the result in this JSON format:
{
  "question": "brief summary of the question",
  "answers": {
    "1": "meaning of response 1",
    "2": "meaning of response 2",
    ...
  }
}"""
        
        user_prompt = f"""Question text: {question_text}
Unique numbers found: {unique_numbers}

Please analyze this survey question and provide a mapping of each number to its meaning."""
        
        response = llm_client.chat(
            system=system_prompt,
            user=user_prompt
        )
        
        # Parse JSON response
        try:
            data = json.loads(response[0])
        except json.JSONDecodeError:
            # Try to clean up the response if it has markdown formatting
            try:
                cleaned_response = response[0].replace('```json', '').replace('```', '').strip()
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # If all parsing fails, return a safe default
                print(f"Failed to parse LLM response: {response[0]}")
                return {"question": question_text, "answers": {}}
        
        return data
    except Exception as e:
        print(f"Error extracting question choices: {e}")
        return {"question": question_text, "answers": {}}


def prep_sentence_node(state: Dict[str, Any], deps: Optional[Any] = None) -> Dict[str, Any]:
    """
    Sentence preparation node that handles the actual text preprocessing
    
    This node:
    1. Extracts column data for the current question
    2. Performs grammar check on all text responses
    3. Routes to appropriate processing function based on question type
    4. Returns processed DataFrame in state
    
    Args:
        state: Current graph state
        deps: Dependencies (contains llm_client)
        
    Returns:
        Updated state with processed results
    """
    llm_client = getattr(deps, "llm_client", LLMClient(model_key="gpt-4.1-mini"))
    
    current_question_id = state.get('current_question_id')
    current_question_type = state.get('current_question_type', 'etc')
    
    print(f"prep_sentence_node: Processing question {current_question_id} with type {current_question_type}")
    
    try:
        # Get column data based on question type
        if current_question_type in ["depend", "depend_pos_neg"]:
            depend_df, text_df = get_column_locations(state, current_question_id, "SENTENCE")
            print(f"Found depend data: {len(depend_df)} rows, text data: {len(text_df)} rows")
        else:
            text_df = get_column_locations(state, current_question_id, "SENTENCE")
            print(f"Found text data: {len(text_df)} rows")
        
        # Get survey context and question summary
        matched_questions = state.get('matched_questions', {})
        question_info = matched_questions[current_question_id]['question_info']
        survey_context = state.get('survey_context', '')
        question_summary = question_info.get('question_summary', question_info.get('question_text', ''))
        
        # Collect all text responses for grammar check
        text_responses = []
        for i in range(len(text_df)):
            # Get the first non-null text value from the row
            row_texts = text_df.iloc[i].dropna().astype(str).tolist()
            if row_texts:
                text_responses.append(row_texts[0].strip())
            else:
                text_responses.append("")
        
        print(f"Performing grammar check on {len(text_responses)} responses...")
        
        # Batch grammar check
        corrected_texts = []
        for i, text in enumerate(text_responses):
            if text and text.strip():
                corrected = grammar_check(llm_client, text, survey_context)
                corrected_texts.append(corrected)
            else:
                corrected_texts.append(text)
            
            if (i + 1) % 50 == 0:
                print(f"  Grammar checked {i + 1}/{len(text_responses)} responses")
        
        print(f"Grammar check completed. Processing by question type: {current_question_type}")
        embed_cols = ['sentence_1', 'sentence_2', 'sentence_3']
        embed = VectorEmbedding()
        
        
        # Route to appropriate processing function
        if current_question_type == "depend_pos_neg":
            # Extract question choices for depend types
            unique_depends = depend_df.iloc[:, 0].dropna().unique().tolist()
            unique_depends = [str(int(x)) for x in unique_depends if pd.notna(x)]
            
            response_dict = extract_question_choices(llm_client, question_summary, unique_depends)
            response_mapping = response_dict.get('answers', {})
            
            result_df = process_depend_pos_neg(
                llm_client=llm_client,
                survey_context=survey_context,
                question_summary=question_summary,
                response_dict=response_mapping,
                depend_df=depend_df,
                text_df=text_df,
                corrected_texts=corrected_texts
            )
            
            
            for col in tqdm(embed_cols, desc="임베딩 진행 중"):
                result_df[f'embed_{col}'] = result_df[col].progress_apply(
                    lambda x: embed.embed(x) if isinstance(x, str) and x.strip() else None
                )
                
        elif current_question_type == "pos_neg":
            result_df = process_pos_neg(
                llm_client=llm_client,
                survey_context=survey_context,
                question_summary=question_summary,
                text_df=text_df,
                corrected_texts=corrected_texts
            )
            for col in tqdm(embed_cols, desc="임베딩 진행 중"):
                result_df[f'embed_{col}'] = result_df[col].progress_apply(
                    lambda x: embed.embed(x) if isinstance(x, str) and x.strip() else None
                )
        elif current_question_type == "depend":
            # Extract question choices for depend types
            unique_depends = depend_df.iloc[:, 0].dropna().unique().tolist()
            unique_depends = [str(int(x)) for x in unique_depends if pd.notna(x)]
            
            response_dict = extract_question_choices(llm_client, question_summary, unique_depends)
            response_mapping = response_dict.get('answers', {})
            
            result_df = process_depend(
                llm_client=llm_client,
                survey_context=survey_context,
                question_summary=question_summary,
                response_dict=response_mapping,
                depend_df=depend_df,
                text_df=text_df,
                corrected_texts=corrected_texts
            )
            for col in tqdm(embed_cols, desc="임베딩 진행 중"):
                result_df[f'embed_{col}'] = result_df[col].progress_apply(
                    lambda x: embed.embed(x) if isinstance(x, str) and x.strip() else None
                )
        else:  # sentence, etc
            result_df = process_sentence(
                llm_client=llm_client,
                survey_context=survey_context,
                question_summary=question_summary,
                text_df=text_df,
                corrected_texts=corrected_texts
            )
            for col in tqdm(embed_cols, desc="임베딩 진행 중"):
                result_df[f'embed_{col}'] = result_df[col].progress_apply(
                    lambda x: embed.embed(x) if isinstance(x, str) and x.strip() else None
                )

        print(f"Processing completed. Result DataFrame shape: {result_df.shape}")
        
        # Update state with results
        return {
            **state,
            'stage2_processed_data': result_df,
            'stage2_question_id': current_question_id,
            'stage2_question_type': current_question_type,
            'stage2_status': 'completed'
        }
        
    except Exception as e:
        print(f"Error in prep_sentence_node: {e}")
        return {
            **state,
            'stage2_error': str(e),
            'stage2_status': 'error'
        }