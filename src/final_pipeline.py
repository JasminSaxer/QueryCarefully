from src.prompts.user_prompts import get_prompt
from src.agents.ollama_agent import OllamaAgent
from src.pipeline import extract_sql_query, SQL2res, correct_sql_query
import logging
import pandas as pd

logger = logging.getLogger(__name__)
def QueryCarefullyPipeline(question, app_stream, embedding_model=None):
    

    options=  {
            'do_not_answer': True,
            'few_shot_seed': 3,
            'few_shot_quna': 3,
            'all_in_systemprompt': True
        }
    
    model = app_stream.session_state["llm_model"]
    
    # Optionally, show temporary text to the user via app_stream if provided
    
    app_stream.write("Getting your prompt ready...")
    
    system_prompt, user_prompt = get_prompt(options, 
                            question, 
                            all_in_systemprompt=options['all_in_systemprompt'], 
                            embedding_model = embedding_model
                            )
    
    logger.info(f'user prompt: {user_prompt}')
    # get the answer from the model
    # SQL_query = generate_one_answer(prompt)
    app_stream.write(f"Running your prompt on the model: {model}...")
    ollama_agent = OllamaAgent(model=model, system_prompt=system_prompt)
    
    # get llm response
    LLM_res = ollama_agent.get_llm_response(user_prompt)
    app_stream.write("Got the response from the model...")
    app_stream.markdown(LLM_res)
    logger.debug(f'Raw LLM Response:\n{LLM_res}')
    
    if 'unanswerable question' in LLM_res:
        app_stream.write("The question is unanswerable. Getting explanation...")

        logger.debug('No SQL query found in the response, asking LLM to expalain to rephrase the question')
        LLM_res = ollama_agent.get_llm_response(
                "Please explain why the question cannot be answered. And ask me to rephrase the question and give me a rephrased question which can be answered.")
        return 'rephrase', LLM_res
    
    
    
    sql_query = extract_sql_query(LLM_res)
    logger.debug(f'Extracted SQL query:\n{sql_query}')

    # check for error and correct
    n_retires = 0
    max_retries = 3
    
    db_res = 'error: not yet run'
    while 'error' in db_res and n_retires < max_retries:
        logger.info('Running SQL query...')
        
        app_stream.write("Getting results from the databse...")

        db_res = SQL2res(sql_query)
        logger.info(f'Got db_res:\n{db_res.keys()}')
        
        # db res has result length, result, and col ordered by
        if 'result' in db_res:
            db_res = db_res['result']
        elif 'result_1000000' in db_res:
            app_stream.write("Result is too long, getting first 1 million rows...")
            db_res = db_res['result_1000000']
        elif 'error' in db_res:
            app_stream.write("There was an error, trying to correct the sql query...")
            logger.info('Error in SQL query execution, trying to correct sql...')
            sql_query, LLM_res = correct_sql_query(db_res, ollama_agent)
            n_retires += 1
            logger.debug(f'SQL_query corrected try {n_retires}: \n{sql_query}')
        else:
            db_res = None
        
        if db_res is not None:
            df_res = pd.DataFrame(db_res)
                
 
    
    app_stream.markdown("Result from the database: ")
    logger.info(f'Got db_res:\n{df_res}')
    app_stream.write(df_res)
    
    logger.info('Getting explanation for the result')
    app_stream.write("Getting explanation for the result...")
    
    if isinstance(df_res, pd.DataFrame):
        df_res_explain = df_res.head(50)
    else:
        df_res_explain = df_res
    
    # logger.info(f'Got db_res_explain:\n{df_res_explain}')
    
    res_explanation = ollama_agent.get_llm_response(
                "Please explain shortly the result from the database. Following is the result from the database (if result is larger than 50, you are only given the first 50, else you are given all results.) " + str(df_res_explain))
    
    app_stream.write("Finished.")
    return 'result', [sql_query, LLM_res, df_res, res_explanation]
