import gzip
import json
import pandas as pd
import re
from tqdm import tqdm 
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def exact_match(pred, gold):
    """
    Calculate the exact_match_accuracy of sql query.
    """

    # Remove all whitespace and convert to lowercase
    pred = re.sub(r'\s+', ' ', pred).lower()
    gold = re.sub(r'\s+', ' ', gold).lower()
        
    return pred == gold

def execution_result(pred_path, gold_path ='data/output/dev_with_dbres.jsonl.gz', verbose=False):
    """
    Calculate the execution_result_accuracy of db result from sql query.
    """
    
    res = {}
    
    if pred_path.endswith('.gz'):
        pred_file = gzip.open(pred_path, 'rt', encoding='utf-8')
    else:
        pred_file = open(pred_path, 'rt', encoding='utf-8')
    
    gold_file = gzip.open(gold_path, 'rt', encoding='utf-8')
    
    for i in tqdm(range(99)):
        try:
            pred = json.loads(pred_file.readline())
        except Exception as e:
            print('Error reading file:', e)
            continue
        
        pred_results = pred['values_model']
        if 'error' in pred_results.keys():
            res_bool = False
            res_reason = 'db error'
            # res_reason = pred_results['error']
            # skip gold line
            gold_file.readline()

        else:
            db_result = pred_results['result']
            db_len = pred_results['result_length']
            col_ordered_by = pred_results['col_ordered_by']
            sql_res_unanswerable = pred['sql_model']
            
            # if db result is None, check if the sql result is unanswerable
            if db_result is None:
                res_bool = False
                if 'question unanswerable' in sql_res_unanswerable:
                    res_reason = 'question unanswerable'
                elif sql_res_unanswerable is None:
                    res_reason = 'no SQL returned'
                else:
                    print('db result none, no reason found!')
                    res_reason = 'unknown'
                
            
            # if db result is a string, check if it is an error
            if isinstance(db_result, str):
                if 'error' in db_result:
                    res_bool = False
                    res_reason = 'db error'
                # skip the line in gold
                gold_file.readline()
            
            # if db result is a list, check if it is correct
            elif isinstance(db_result, list):
                
                gold_row = json.loads(gold_file.readline())
                
                # check if correct gold to pred
                if gold_row['question'] != pred['question']:
                    gold_row = json.loads(gold_file.readline())
                    if gold_row['question'] != pred['question']:
                        print('Error question not the same!!')
                        return -1
                    
                gold_res = gold_row['db_result']
                
            
                # Ensure gold_res and db_result are lists of dicts or lists
                if not isinstance(gold_res, (list, tuple)):
                    gold_res = [gold_res]
                    
                gold_res = pd.DataFrame(gold_res)
                gold_res = gold_res.apply(pd.to_numeric, errors='ignore')
                
             
                pred_res = pd.DataFrame(db_result)
                pred_res = pred_res.apply(pd.to_numeric, errors='ignore')
                
                
                if len(gold_res) != len(pred_res):
                    res_bool = False
                    res_reason = 'incorrect result'
                    
                elif gold_res.equals(pred_res) or gold_res.sort_values(by=gold_res.columns.tolist()).reset_index(drop=True).equals(pred_res.sort_values(by=pred_res.columns.tolist()).reset_index(drop=True)):
                    res_bool = True
                    res_reason = 'result exact match'
                    # check if sql exact match 
                    if exact_match(pred['sql_model'], gold_row['query']):
                        res_reason = 'sql exact match'
                    
                elif gold_res.values.tolist() == pred_res.values.tolist():
                    res_bool = True
                    res_reason = 'correct (ignoring column names)'  
                    if verbose:
                        print('Correct (ignoring column names)')
      
                    
                elif gold_res.drop(columns=['id'], errors='ignore').equals(pred_res.drop(columns=['id'], errors='ignore')):
                    res_bool = True
                    res_reason = 'correct (ignoring id column)'       
                    if verbose:
                        print('Correct (ignoring id column)')
                
                else:
                    res_bool = False
                    res_reason = 'incorrect result'
                
                # check the reason for false result
                # if not res_bool:
                #     # not same columns
                #     if set(gold_res.columns.tolist()) != set(pred_res.columns.tolist()):
                #         # Calculate the percentage of predicted columns that are in gold columns
                #         pred_cols = set(pred_res.columns.tolist())
                #         gold_cols = set(gold_res.columns.tolist())
                #         common_cols = pred_cols.intersection(gold_cols)
                #         if len(gold_cols) == 0 and len(pred_cols) >=0:
                #             # print('gold res no columns? : ', gold_row['db_result'])
                #             res_reason = 'gold result have no columns, but predicted has'
                #         else:
                #             res_reason = f'predicted result columns are subset of gold result columns'
                                
                        
                #     elif len(gold_res) != len(pred_res):
                #         res_reason = 'length of result different'
                #         # is_subset = all(item in pred_res.values.tolist() for item in gold_res.values.tolist())
                        # if is_subset:
                        #     res_reason = 'gold result is subset of predicted result'
                        # elif all(item in gold_res.values.tolist() for item in pred_res.values.tolist()):
                        #     res_reason = 'predicted result is subset of gold result'
                        # else:
                        #     res_reason = 'unknown specific reason'
                
                # except Exception as e:
                #     print('Error comparing res:', e)
                #     display(gold_res)
                #     display(pred_res)
                    
            else:
                # skip the line
                gold_file.readline()

        # put in the result
        res[pred['question']] = {
            'res_bool': res_bool,
            'res_reason': res_reason
        }
    
    return res



def unans_question(pred_path):
    if pred_path.endswith('.gz'):
        pred_file = gzip.open(pred_path, 'rt', encoding='utf-8')
    else:
        pred_file = open(pred_path, 'rt', encoding='utf-8')
    
    res = {}
    # Find the number of lines (questions) in the pred_file
    if pred_path.endswith('.gz'):
        with gzip.open(pred_path, 'rt', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        pred_file.seek(0)
    else:
        with open(pred_path, 'rt', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        pred_file.seek(0)

    for i in tqdm(range(total_lines)):
        try:
            pred = json.loads(pred_file.readline())
        except Exception as e:
            print('Error reading file:', e)
            continue
        
        
        
        sql_res_unanswerable = pred['sql_model']
        db_results = pred['values_model']
        lm_response = pred['lm_response_model']
                

        if 'question unanswerable' in sql_res_unanswerable or 'unanswerable question' in sql_res_unanswerable:
            res_bool = True
            res_reason = 'question unanswerable'
            db_result = None
        
        else:
            res_bool = False
            if db_results is None:
                print(sql_res_unanswerable)
                print(db_results)
                print(lm_response)
                
                
            if 'error' in db_results.keys():
                res_reason = 'db error'
                db_result = db_results['error']

            elif 'result' not in db_results.keys():
                db_result = db_results['result_10000']
                res_reason = 'db result'
                
            else:
                db_result = db_results['result']     
                if db_result is None:
                    res_reason = 'db result none'
                elif db_result == []:
                    res_reason = 'db result empty'
                elif db_result: 
                    res_reason = 'db result'
                else:
                    res_reason = 'unknown'
                
        res[pred['question']] = {
            'res_bool': res_bool,
            'res_reason': res_reason, 
            'db_result': db_result, 
            'sql_model': pred['sql_model'], 
            'llm_response_model': pred['lm_response_model']
        }

    return res