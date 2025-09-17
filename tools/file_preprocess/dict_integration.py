import json


def data_integration(questions, question_data_match):
    """
    Integrate two input dictionaries into a single output dictionary.

    Args:
        input_dict_1 (dict): The first input dictionary.
        input_dict_2 (dict): The second input dictionary.

    Returns:
        dict: A new dictionary containing all key-value pairs from both input dictionaries.
              If there are overlapping keys, values from input_dict_2 will overwrite those from input_dict_1.
    """
    
    if isinstance(questions, str):
        questions = json.loads(questions)
    if isinstance(question_data_match, str):
        question_data_match = json.loads(question_data_match)
        
    integrated_map = {}
    
    for question in questions:
        question_number = question['open_question_number']

        integrated_map[question_number] = {
            'question_info': {
                'question_text': question['question_text'],
                'question_type': question['question_type'],
                'related_question': question['question_that_is_related']
            },
            'matched_columns': question_data_match.get(question_number, [])
        }
    
    return integrated_map