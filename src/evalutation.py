import gzip
import json
import pandas as pd
import re
from tqdm import tqdm 


def execution_success_rate(pred_values):
    pred_values_no_error = [i for i in pred_values if isinstance(i, list)]
    return len(pred_values_no_error)/len(pred_values)
    

def exact_match_accuracy(pred, gold):
    """
    Calculate the exact_match_accuracy of sql query.
    """
    if len(pred) != len(gold):
        return 'Values of pred and gold are not the same length!'
    
    counter_corr = 0
    for p, g in zip(pred, gold):
        
        if p is None:
            continue
        # Remove all whitespace and convert to lowercase
        p = re.sub(r'\s+', ' ', p).lower()
        g = re.sub(r'\s+', ' ', g).lower()
        
        if p == g: 
            counter_corr += 1
    
    return counter_corr / len(gold)


def execution_result_accuracy(pred, gold_path ='data/output/dev_with_dbres.jsonl.gz', verbose=False):
    """
    Calculate the execution_result_accuracy of db result from sql query.
    """
    i = 0
    counter_corr = 0
    res_reason_list = []
    with gzip.open(gold_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total = len(pred)):
            if i >= len(pred):
                break
            
            if isinstance(pred[i], list):
                try:

                    line_res = json.loads(line)
                    gold_res = line_res['db_result']
                    pred_res = pred[i]
                
                    gold_res = pd.DataFrame(gold_res)
                    pred_res = pd.DataFrame(pred_res)
                    gold_res = gold_res.apply(pd.to_numeric, errors='ignore')
                    pred_res = pred_res.apply(pd.to_numeric, errors='ignore')
                    
                    if len(gold_res) != len(pred_res):
                        res_reason = 'different result'
                    elif gold_res.equals(pred_res) or gold_res.sort_values(by=gold_res.columns.tolist()).reset_index(drop=True).equals(pred_res.sort_values(by=pred_res.columns.tolist()).reset_index(drop=True)):
                        counter_corr += 1
                        res_reason = 'correct'
                    elif gold_res.values.tolist() == pred_res.values.tolist():
                        counter_corr += 1
                        res_reason = 'correct (ignoring column names)'  
                        if verbose:
                            print('Correct (ignoring column names)')
                      
                            display(gold_res)
                            display(pred_res)
                        
                    elif gold_res.drop(columns=['id'], errors='ignore').equals(pred_res.drop(columns=['id'], errors='ignore')):
                        counter_corr += 1
                        res_reason = 'correct (ignoring id column)'       
                        if verbose:
                            print('Correct (ignoring id column)')
                            display(gold_res)
                            display(pred_res)
                    else:
                        res_reason='different result'

   
                except Exception as e:
                    print('Error:', e)
                    display(gold_res)
                    display(pred_res)
            
            else:
                res_reason = pred[i]
                
            res_reason_list.append(res_reason)
            
            i += 1
    
    return counter_corr / len(pred), res_reason_list

def unanswerable_question_detection(pred, gold='question unanswerable'):
    
    counter_corr = 0
    for p in pred:
        if p == gold: 
            counter_corr += 1
    
    return counter_corr / len(pred)