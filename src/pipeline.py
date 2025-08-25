from src.agents.ollama_agent import OllamaAgent
from src.agents.postgres_agent import PostgresAgent
from src.prompts.user_prompts import get_correct_sql_prompt
import logging
import re


# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def NL2SQL(prompt,system_prompt='', model='', check_for_error=False, check_return_sql=False):
    
    logger.info('Running prompt...')
    # get the answer from the model
    # SQL_query = generate_one_answer(prompt)
    ollama_agent = OllamaAgent(model=model, system_prompt=system_prompt)

    # get llm response
    LLM_res = ollama_agent.get_llm_response(prompt)
    logging.debug(f'Raw LLM Response:\n{LLM_res}')
    
    # if unanserable question, return it..
    if 'unanswerable question' in LLM_res:
        logger.info('Unanswerable question detected')
        return 'unanswerable question', LLM_res, None
    
    # else get sql_query
    sql_query = extract_sql_query(LLM_res)
    
    
    # if no sql query extraced and no unanswerable question in LLM response, ask the LLM to return a SQL query or 'unanswerable question'
    if sql_query is None:
        logger.info('No SQL query found in the response')

        # tell llm to return an sql query or 'cannot be answered'
        if check_return_sql:
            logger.info('Asking LLM to return a SQL query or "unanswerable question"')
            LLM_res = ollama_agent.get_llm_response(
                "Please return a SQL query or 'unanswerable question' if the question cannot be answered with an SQL Query on the database.")
            print('SQL query:', LLM_res)
            
            # check again if the response is a SQL query or 'unanswerable question'
            if 'unanswerable question' in LLM_res:
                return 'unanswerable question', LLM_res, None
            else:
                sql_query = extract_sql_query(LLM_res)
                if sql_query is None:
                    logger.debug('No SQL query found in the response')
                    return None, LLM_res, None
    
    # run sql query and check for error and correct
    if check_for_error:
        logger.info('Checking for SQL query execution error')
        n_retires = 0
        max_retries = 3
        
        db_res = 'error: not yet run'
        while 'error' in db_res and n_retires < max_retries:
            logger.info('Running SQL query...')
            db_res = SQL2res(sql_query)
            
            if 'error' in db_res:
                logger.info('Error in SQL query execution, trying to correct sql...')
                logger.debug(f'Error message: {db_res}')
                sql_query, LLM_res = correct_sql_query(db_res, ollama_agent)
                n_retires += 1
            
                logger.debug(f'SQL_query corrected try {n_retires}: \n{sql_query}')
    
    
    # just get the result without correcting error
    else:
        if sql_query is not None:
            # run the SQL query
            logger.info('Running SQL query...')
            db_res = SQL2res(sql_query)
        else:
            db_res = None
            logger.debug('No SQL query found in the response')

    return sql_query, LLM_res, db_res

def correct_sql_query(db_result, ollama_agent):
    logger.debug('Error in SQL query execution')
    
    correct_sql_prompt = get_correct_sql_prompt(db_result)
    # try to correct the SQL query
    corrected_llm_response = ollama_agent.get_llm_response(correct_sql_prompt)
    sql_query = extract_sql_query(corrected_llm_response)
    return sql_query, corrected_llm_response
    
            
def SQL2res(SQL_query):
    # test the result
    logger.debug(f'Running SQL query: {SQL_query}')
    agent = PostgresAgent()

    # return as dictionary..
    db_result = agent.run_query(SQL_query)
    # logger.debug(f'Result: {db_result}')

    agent.close()
    
    return db_result


def extract_sql_query(LLM_res):
    # process the SQL query
    SQL_query = LLM_res
    if 'sql' in SQL_query:
        logging.debug('Extract SQL Query from the response')
        # Extract the SQL query between ```sql and ```
        match = re.search(r'```sql(.*?)```', SQL_query, re.DOTALL)
        if match:
            SQL_query = match.group(1).strip()
    if '```' in SQL_query:
        logging.debug('Extract SQL Query from the response')

        # Extract the SQL query between ``` and ```
        match = re.search(r'```(.*?)```', SQL_query, re.DOTALL)
        if match:
            SQL_query = match.group(1).strip()
            
    # Check if the SQL query contains 'SELECT' and ends with ';'
    if 'SELECT' in SQL_query:
        logging.debug('Extracting SQL query starting with SELECT and ending with ;')
        # Extract the part of the query starting with SELECT and ending with ;
        match = re.search(r'(SELECT.*?;)', SQL_query, re.DOTALL | re.IGNORECASE)
        if match:
            SQL_query = match.group(1).strip()
            
    else:
        logger.debug('No SQL query found in the response')
        return None
    
    return SQL_query 

