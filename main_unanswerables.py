from src.pipeline import NL2SQL, SQL2res
from src.prompts.user_prompts import get_prompt
from src.agents.ollama_agent import is_model_available, pull_ollama_model
import pandas as pd
from tqdm import tqdm 
from time import time
import os
import simplejson as json

tqdm.pandas()
    
def get_llm_response(question, model, option, question_embedding, all_in_systemprompt=False, check_for_error=False):

    system_prompt, user_prompt = get_prompt(option, 
                    question, 
                    question_embedding=question_embedding, 
                    all_in_systemprompt=all_in_systemprompt
                    )
    
    # Get the result from the pipeline
    sql_query, LM_response, db_res = NL2SQL(user_prompt, system_prompt=system_prompt, model=model, check_for_error=check_for_error)
    
    return sql_query, LM_response, db_res

def run_options():
    
    df = pd.read_pickle('data/addtional/unanswerables_embeddings.pkl')
    
    # check if the model is available
    model = 'llama3.3:70b'
    is_model_available(model)
    
    # for specific options
    options = [
        # {
        #     'experiment_name': 'base_schema_in_systemprompt',
        #     'do_not_answer': False,
        #     'few_shot_seed': None,
        #     'few_shot_quna': None, 
        #     'all_in_systemprompt': True
        # },
        # # {
        # #     'experiment_name': 'base_schema_in_userprompt',
        # #     'do_not_answer': False,
        # #     'few_shot_seed': None,
        # #     'few_shot_quna': None, 
        # #     'all_in_systemprompt': False,
        # # },
            
        # {
        #     'experiment_name': 'dna',
        #     'do_not_answer': True,
        #     'few_shot_seed': None,
        #     'few_shot_quna': None,
        #     'all_in_systemprompt': True
        # },
        
        # {
        #     'experiment_name': 'fs_seed1',
        #     'do_not_answer': True,
        #     'few_shot_seed': 1,
        #     'few_shot_quna': None,
        #     'all_in_systemprompt': True
        # },
        
        # {
        #     'experiment_name': 'fs_seed3',
        #     'do_not_answer': True,
        #     'few_shot_seed': 3,
        #     'few_shot_quna': None,
        #     'all_in_systemprompt': True
        # },
        
        # {
        #     'experiment_name': 'fs_seed5',
        #     'do_not_answer': True,
        #     'few_shot_seed': 5,
        #     'few_shot_quna': None,
        #     'all_in_systemprompt': True
        # }, 
        
        #       {
        #     'experiment_name': 'fs_quna1',
        #     'do_not_answer': True,
        #     'few_shot_seed': None,
        #     'few_shot_quna': 1,
        #     'all_in_systemprompt': True
        # },
        
        #  {
        #     'experiment_name': 'fs_quna3',
        #     'do_not_answer': True,
        #     'few_shot_seed': None,
        #     'few_shot_quna': 3,
        #     'all_in_systemprompt': True
        # },
         
        #   {
        #     'experiment_name': 'fs_quna5',
        #     'do_not_answer': True,
        #     'few_shot_seed': None,
        #     'few_shot_quna': 5,
        #     'all_in_systemprompt': True
        # },
        # {
        #     'experiment_name': 'fs_quna3_seed3',
        #     'do_not_answer': True,
        #     'few_shot_seed': 3,
        #     'few_shot_quna': 3,
        #     'all_in_systemprompt': True
        # },
        {
            'experiment_name': 'fs_quna5_seed5',
            'do_not_answer': True,
            'few_shot_seed': 5,
            'few_shot_quna': 5,
            'all_in_systemprompt': True
        },
        {
            'experiment_name': 'fs_quna5_seed5_check_error',
            'do_not_answer': True,
            'few_shot_seed': 5,
            'few_shot_quna': 5,
            'all_in_systemprompt': True, 
            'check_for_error': True
        },
        
    ]
    
    
    for option in options:
        print('Running with options:', option)
        
        folder = f'data/output/{model}/Qunans/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        path = f"qunans_{option['experiment_name']}.pkl"
        save_path = os.path.join(folder, path)
        if os.path.exists(save_path):
            print('File already exists, skipping...')
            continue
        # save the prompt 
        system_prompt, user_prompt = get_prompt(option, 
                            df.iloc[0]['question'], 
                            question_embedding=df.iloc[0]['question_embedding_Alibaba-NLP/gte-Qwen2-1.5B-instruct'], 
                            all_in_systemprompt=option['all_in_systemprompt']
                            )
        
        # save prompts to file
        with open(os.path.join(folder, f"system_prompt_{option['experiment_name']}.txt"), "w") as f:
            f.write(system_prompt)
        with open(os.path.join(folder, f"user_prompt_{option['experiment_name']}.txt"), "w") as f:
            f.write(user_prompt)
        
       
        if os.path.exists(save_path.replace('.pkl', '.jsonl')):
            print('JSONL file already exists, ')
            'find the last question in the jsonl file'
            with open(save_path.replace('.pkl', '.jsonl'), 'r') as jsonl_file:
                for line in jsonl_file:
                    pass
                last_line = line
                last_row = json.loads(last_line)
                last_question= last_row['question']
                print('Last question:', last_question)
            
            get_result = False
        else:
            last_question = None
            get_result = True
        # run the options
        start_t = time()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            
            if last_question is not None:
                if row['question'] == last_question:
                    get_result = True
                    continue
            
            if get_result:
                sql_res, LLM_response, db_result = get_llm_response(
                    row['question'], 
                    model, 
                    option, 
                    question_embedding=row['question_embedding_Alibaba-NLP/gte-Qwen2-1.5B-instruct'], 
                    all_in_systemprompt=option['all_in_systemprompt'], 
                    check_for_error=option.get('check_for_error', False)
                )

                # Save each row as a JSONL line
                with open(save_path.replace('.pkl', '.jsonl'), 'a') as jsonl_file:
                    row_dict = row.to_dict()
                    row_dict.pop('question_embedding_Alibaba-NLP/gte-Qwen2-1.5B-instruct', None)
                    row_dict['sql_model'] = sql_res
                    row_dict['lm_response_model'] = LLM_response
                    row_dict['values_model'] = db_result
                    jsonl_file.write(json.dumps(row_dict) + '\n')
        
        # all at once
        # df[['sql_model', 'lm_response_model', 'values_model']] = df.progress_apply(
        #     lambda x: get_llm_response(x['question'], model, option, question_embedding=x['question_embedding']), axis=1, result_type='expand'
        # )

        # df.to_pickle(save_path)
        end_t = time()
        print('Time taken:', end_t - start_t)
        
def testing():
    df = pd.read_pickle('data/addtional/unanswerables_embeddings.pkl')
    
if __name__ == "__main__":

    # testing()
    
    run_options()
  