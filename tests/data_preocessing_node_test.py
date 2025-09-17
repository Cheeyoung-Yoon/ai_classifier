# %%
import sys
import os
sys.path.append(os.path.abspath("/home/cyyoon/test_area/ai_text_classification/2.langgraph"))
import json
    # get data 
import pandas as pd


with open("/home/cyyoon/test_area/ai_text_classification/2.langgraph/tests/debug_states/print_20250916_141551_740306_AFTER_STAGE_1_MEMORY_FLUSH.json", "r") as f:
    state = json.load(f)
# %%
def get_column_locations():
    
    
    df_path = state['dataframe']
    df_path = './data/suv/raw_data.csv'
    df= pd.read_csv(df_path)
    matched_cols = sample_mock[QID]['matched_columns']

    df_col_loc_list = [df.columns.get_loc(col) for col in matched_cols]
    # if node_type is depend, add min -1 element. 
    partition = df[df.columns[df_col_loc_list]]
    if node_type == "DEPEND":
        
        
        depend_col = [min(df_col_loc_list) - 1]
        depend_col = df[df.columns[depend_col]]
        if depend_col.isna().sum().sum() == len(depend_col):
            depend_col = [min(df_col_loc_list) - 2]
            depend_col = df[df.columns[depend_col]]
        
        return depend_col , partition

    return partition


QTYPE_MAP = {
    "concept": "WORD",
    "img": "WORD",
    "depend": "SENTENCE",
    "depend_pos_neg": "SENTENCE",
    "pos_neg": "SENTENCE",
    "etc": "ETC",
}

sample_mock = state['matched_questions']
for QID in sample_mock.keys():
    qtype = sample_mock[QID]['question_info']['question_type']

    node_type = QTYPE_MAP.get(qtype, "__END__")
    # print(f"QID: {QID}, Type: {qtype} -> Node Type: {node_type}")
    
    print(f"QID: {QID}, Question Type: {qtype}")
    if node_type == "WORD":
        part = get_column_locations()
        print(f"  Word Node for QID {QID}")
    elif node_type == "DEPEND":
        depend , part = get_column_locations()
        print(f"  Depend Node for QID {QID}")
    elif node_type == "POS_NEG":
        part = get_column_locations()
        print(f"  Pos/Neg Node for QID {QID}")
    elif node_type == "ETC":
        part = get_column_locations()
        print(f"  Etc Node for QID {QID}")

# %%

# if word: no llm just pass to embedding node 



## pass for now

# in sentence, there is depend, pos_neg, and depend_pos_neg. 
# for depend, add the depeding column as the context.
# if posneg add a prompt to control the pos/neg
# if depend_pos_neg, add both the depending column and pos/neg prompt.

# if depend: pass part and depend columns to llm to get answer
# question text + depend column + part columns -> llm -> answer
# depend -> understand the code -> extract relevant code snippets
# part + depend understadning schema -> define relvant, svc, and automic setences. (SVC can be skiiped)

for QID in sample_mock.keys():
    qtype = sample_mock[QID]['question_info']['question_type']

    node_type = QTYPE_MAP.get(qtype, "__END__")
    # print(f"QID: {QID}, Type: {qtype} -> Node Type: {node_type}")
    
    print(f"QID: {QID}, Question Type: {qtype}")
    if node_type == "DEPEND":
        depend , part = get_column_locations()
        print(f"  Depend Node for QID {QID}")






# %%
import os 
import sys
sys.path.append("/home/cyyoon/test_area/ai_text_classification/2.langgraph")

from io_layer.llm.client import LLMClient

llm_client = LLMClient(model_key="gpt-4.1-nano")

system_prompt = """
Based on the text given, understand the answer labels of the multiple choice question.
don't add any opinions or extra information.
return all the choices with their explanations in json format.
summarize the explination for each answer choice. so that it has clear explination of the answer choice.
Return in the following JSON format:
{
    "question": summary of the question,
  "answers": {
    "1": "explanation for answer 1",
    "2": "explanation for answer 2", 
    "3": "explanation for answer 3",
    ...
    "N": "explanation for answer N"
  }
}

Use actual integer keys as strings and provide clear explanations for each answer choice.
"""
QID = "문17-1"

question_text= state['matched_questions'][QID]['question_info']['question_text']
user_prompt = f"""
question: {question_text}
"""

response = llm_client.chat(
    system = system_prompt,
    user = user_prompt
)

question_summary = json.loads(response[0])['question']
response_dict = json.loads(response[0])['answers']



# %%

part["merged"] = part.fillna("").astype(str).agg(" ".join, axis=1)
df = part[["merged"]]
# %%
sentence_split = """You are a survey response interpreter.  
You will receive input in JSON format containing:  
1) question: the original survey question explanation  
2) sub_explanation: additional clarification of the question’s intent  
3) answer: the respondent’s free-text answer  

### Rules
1. **Nuance reference**: The “question” and “sub_explanation” are only for context. Do not include them in the final output.  
2. **Question matching judgment**:  
   - Set `"matching_question": true` unless the answer is clearly irrelevant or meaningless.  
   - Even short or abstract answers like *“trustworthy”*, *“it feels reliable”*, *“good”*, *“satisfying”* should be treated as `true`.  
   - Set `"matching_question": false` only if the answer is irrelevant, empty, meaningless tokens, or pure emotional expression with no semantic relation (e.g., “Wow!”, “I’m so excited”).  
   - Ignore filler tokens such as `merged`, `nan`, `NaN`, `None`, `NULL`, or `Name: 0, dtype: object` before evaluation.  
3. **Atomic sentence split**: If the answer contains multiple semantic units, split it into at most 3 atomic sentences (Subject–Verb–Complement based).  
4. **S/V/C keyword extraction**: For each atomic sentence, extract core keywords:  
   - S = subject (main entity)  
   - V = verb (main action/state)  
   - C = complement/object (if any)  
   - Only extract essential words, not particles or function words.  
5. **Output format**: Return the result strictly in the following JSON format (schema stays in Korean):  
6. **Pos/Neg classification**: If the question is about sentiment (positive/negative), classify the answer accordingly with sub_explanation and set `"pos.neg"` to `"POSITIVE"`or `"NEGATIVE"`.
```json
{
  "matching_question": true/false,
  "pos.neg": "POSITIVE"/"NEGATIVE",
  "automic_sentence": ["문장1", "문장2", "문장3"],
  "SVC_keywords": {
    "S": [],
    "V": [],
    "C": []
  }
}

"""
result = pd.DataFrame()
for i in range(15,25):
    sample = response_dict[str(int(depend.iloc[i]))] 


    user_prompt = f"""
    question: {question_summary}
    sub_explination: {sample} 
    answer : {df.iloc[i].values[0].strip()}
    """


    response2 = llm_client.chat(
        system = sentence_split,
        user = user_prompt
    )

    data = json.loads(response2[0])
    df_result = pd.DataFrame({
        'id': [i],
        'pos.neg': [data.get('pos.neg', None)],
        'matching_question': [data['matching_question']],
        'sentence_1': [data['automic_sentence'][0] if len(data['automic_sentence']) > 0 else None],
        'sentence_2': [data['automic_sentence'][1] if len(data['automic_sentence']) > 1 else None], 
        'sentence_3': [data['automic_sentence'][2] if len(data['automic_sentence']) > 2 else None],
        'S': [', '.join(data['SVC_keywords']['S'])],
        'V': [', '.join(data['SVC_keywords']['V'])],
        'C': [', '.join(data['SVC_keywords']['C'])]
    })
    result = pd.concat([result, df_result], ignore_index=True)
# %%
# if other :
# part -> define relvant, svc, and automic setences. (SVC can be skiiped)


sentence_split = """You are a survey response interpreter.  
You will receive input in JSON format containing:  
1) question: the original survey question explanation  
2) answer: the respondent’s free-text answer  

### Rules
1. **Nuance reference**: The “question” and “sub_explanation” are only for context. Do not include them in the final output.  
2. **Question matching judgment**:  
   - Set `"matching_question": true` unless the answer is clearly irrelevant or meaningless.  
   - Even short or abstract answers like *“trustworthy”*, *“it feels reliable”*, *“good”*, *“satisfying”* should be treated as `true`.  
   - Set `"matching_question": false` only if the answer is irrelevant, empty, meaningless tokens, or pure emotional expression with no semantic relation (e.g., “Wow!”, “I’m so excited”).  
   - Ignore filler tokens such as `merged`, `nan`, `NaN`, `None`, `NULL`, or `Name: 0, dtype: object` before evaluation.  
3. **S/V/C keyword extraction**: For each atomic sentence, extract core keywords:  
   - S = subject (main entity)  
   - V = verb (main action/state)  
   - C = complement/object (if any)  
   - Only extract essential words, not particles or function words.  
4. **Output format**: Return the result strictly in the following JSON format (schema stays in Korean):  

```json
{
  "matching_question": true/false,
  "automic_sentence": ["문장1", "문장2", "문장3"],
  "SVC_keywords": {
    "S": [],
    "V": [],
    "C": []
  }
}

"""


# if etc: 
# pass for now. 