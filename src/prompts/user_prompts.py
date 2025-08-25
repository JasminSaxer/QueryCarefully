import sys
import os

try:
    from src.few_shot_nearest.few_shot_nearest import question_embedding

except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'few_shot_nearest'))
    from few_shot_nearest import question_embedding


do_not_answer = """# Rules to not answer unanswerable questions
If a natural language question is not translatable into a SQL statement do not generate an SQL statement but return “unanswerable question”.  
If the table, column or value are not present in the database do not generate an SQL statement but return “unanswerable question”. 
If the context is not clear do not generate an SQL statement but return “unanswerable question”. 
If there is more than one possible table, column, value or operator possible for the question and it is not clear, which one is meant, do not generate an SQL statement but return “unanswerable question”.

"""


system_prompt_start = """Act as an expert in natural language to SQL translation for the OncoMX relational database, part of the Science Benchmark project. 
Your task is to convert a user’s natural language question into a syntactically correct PostgreSQL query. 
Only return the SQL query without explanation.

"""

def get_few_shot_seed(question, n, answerable, embedding, question_embedder=None):
    if question_embedder is None:
        question_embedder = question_embedding('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
    nearest = question_embedder.get_nearest_neighbors(question, answerable, n, embedding)
    
    if answerable:
        prompt_few_shot = "# Example answerable question\n"
        for i, row in nearest.iterrows():
            prompt_few_shot += f"[Q]: {row['question']}\n"
            prompt_few_shot += f"[SQL]: '''{row['query']}'''\n\n"
        prompt_few_shot += "\n"
    else:
        prompt_few_shot = "# Example unanswerable question\n"
        for i, row in nearest.iterrows():
            prompt_few_shot += f"[Q]: {row['question']}\n"
            prompt_few_shot += f"[SQL]: '''unanswerable question'''\n\n"
        prompt_few_shot += "\n"
        
    return prompt_few_shot

    
def get_sql_schema():
    """
    Load the SQL schema from a file.
    """
    with open('data/oncomx/readable_schema.txt', 'r') as file:
        sql_schema = file.read()
    sql_schema = '# Schema\n' + sql_schema
    return sql_schema


    
def get_prompt(option, question, question_embedding=None, all_in_systemprompt=False, embedding_model=None):
    """
    Generate a SQL prompt based on the given options and question.

    Args:
        option (dict): A dictionary containing the following keys:
            - 'rules' (bool): Whether to include the rules in the prompt.
            - 'do_not_answer' (bool): Whether to include the "do not answer" guidelines in the prompt.
            - 'few_shot_seed' (int): Number of few-shot examples to include (1 or 3).
            - 'few_shot_quna' (int): Number of question unanswerable few-shot examples to include (1 or 3).
        question (str): The natural language question to be converted into a SQL query.

    Returns:
        str: The generated SQL prompt.
    """
    
    sql_schema = get_sql_schema()
    
    base_prompt = f"""# Return the SQL for the following Question\n[Q]: {question}\n[SQL]:'''"""
    user_prompt = ""
    # SYSTEM PROMPT
    system_prompt = system_prompt_start 
    # add do not answer rules
    if option['do_not_answer']:
        system_prompt += do_not_answer
    
    # add provided with 
    you_are_provided_with= """# You are provided with:\n - database schema\n"""

    # get few shot examples
    if option['few_shot_seed']:
        few_shot_prompt = get_few_shot_seed(question, option['few_shot_seed'], answerable=True, embedding=question_embedding, question_embedder=embedding_model)
        you_are_provided_with += "- Examples of answerable natural language question and their SQL queries\n"

        # print('---', few_shot_prompt, '---')
        
    if option['few_shot_quna']:
        few_shot_quna = get_few_shot_seed(question, option['few_shot_quna'], answerable=False, embedding=question_embedding, question_embedder = embedding_model)
        you_are_provided_with+= '- Examples of unanswerable natural language question\n'

    system_prompt += you_are_provided_with + '\n'
    
    # ADDITIONAL INFO
    # add schema after rules
    additonal_info = sql_schema
    
    # add few shot examples
    if option['few_shot_seed']:
        additonal_info += few_shot_prompt
        
    if option['few_shot_quna']:
        additonal_info += few_shot_quna
    
    
    # WHERE TO ADD THE ADDITIONAL INFO
    # add everything into the system prompt except the question
    if all_in_systemprompt:
        system_prompt = system_prompt + additonal_info
    else:
        user_prompt = additonal_info
        
        
    # add base prompt in the end
    user_prompt += base_prompt
    
    return system_prompt, user_prompt

def get_correct_sql_prompt(db_result):
    """
    Generate a prompt to correct the SQL query based on the error message.

    Args:
        db_result (str): The error message from the SQL query execution.

    Returns:
        str: The generated prompt to correct the SQL query.
    """
    return f"Please correct the SQL query based on the following error message: {db_result}"



if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_pickle('data/output/dev_fixed_2cols_embeddings.pkl')

    option = {
            'do_not_answer': True,
            'few_shot_seed': 3,
            'few_shot_quna': 3
        }
    
    system_prompt, user_prompt = get_prompt(option, 
                                            df.iloc[0]['question'], 
                                            question_embedding=df.iloc[0]['question_embedding'],
                                            all_in_systemprompt=True)
    
    print('--- SYSTEM PROMPT ---')
    print(system_prompt)
    print('--- USER PROMPT ---')
    print(user_prompt)
