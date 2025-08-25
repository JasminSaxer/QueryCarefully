from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np

class question_embedding:
    def __init__(self, model_name):
        self.model_name = model_name
        # self.model = SentenceTransformer(model_name)
    
    def encode(self, questions):
        self.model = SentenceTransformer(self.model_name)
        return self.model.encode(questions, normalize_embeddings=True)

    def get_nearest_neighbors(self, question, answerable, n=1, embedding = None):
        
        if embedding is not None: 
            question_embedding = [embedding]
        else:
            question_embedding = self.encode([question])
        
        # get the seed questions 
        if answerable:
            df_seed_embeds = get_seed_questions(self.model_name)
        if not answerable:
            df_seed_embeds = get_seed_questions_unanswerable(self.model_name)
            
        # Calculate cosine similarity
        embeddings = np.vstack(df_seed_embeds[f'question_embedding_{self.model_name}'].to_numpy())
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get the indices of the top n most similar questions
        if not answerable:
            # get the top n most similar questions without the first one
            nearest_indices = similarities.argsort()[-n-1:][::-1][1:]
        else:
            nearest_indices = similarities.argsort()[-n:][::-1]
        
        if 'query' in df_seed_embeds.columns:
            questions_list = df_seed_embeds.loc[nearest_indices, ['question', 'query']]
        else:
            questions_list = df_seed_embeds.loc[nearest_indices, ['question']]
        
        return questions_list
    
    
def get_seed_questions(model_name):
    # Load the seed questions from a file
    
    path_seed_embeddings = 'data/addtional/seed_embeddings.pkl'
    col_name = 'question_embedding_' + model_name
    if os.path.exists(path_seed_embeddings):
        df_seed_emebds = pd.read_pickle(path_seed_embeddings)
        if col_name in df_seed_emebds.columns:
            return df_seed_emebds

    question_embedder = question_embedding(model_name)
    path_seed = 'data/oncomx/seed.json'
    df_seed_emebds = pd.read_json(path_seed)
    df_seed_emebds[col_name] = list(question_embedder.encode(df_seed_emebds['question']))
    # save to pickle
    df_seed_emebds[['question', 'query', col_name]].to_pickle(path_seed_embeddings)
    
    return df_seed_emebds[['question', 'query', col_name]]

def get_seed_questions_unanswerable(model_name):
    # Load the seed questions from a file
    
    path_seed_embeddings = 'data/addtional/unanswerables_embeddings.pkl'
    col_name = 'question_embedding_' + model_name
    if os.path.exists(path_seed_embeddings):
        df_seed_emebds = pd.read_pickle(path_seed_embeddings)
        if col_name in df_seed_emebds.columns:
            return df_seed_emebds

    question_embedder = question_embedding(model_name)
    path_seed = 'data/addtional/unanswerable_questions.csv'
    df_seed_emebds = pd.read_csv(path_seed, sep = '\t')
    df_seed_emebds[col_name] = list(question_embedder.encode(df_seed_emebds['question']))
    # save to pickle
    df_seed_emebds[['question',col_name]].to_pickle(path_seed_embeddings)
    
    return df_seed_emebds[['question', col_name]]

if __name__ == "__main__":
    
    # questions = question_embedding('Alibaba-NLP/gte-Qwen2-1.5B-instruct').get_nearest_neighbors('Show me all disease mutations with ref_aa E', 5, answerable=False)
    
    unansquestion = question_embedding('Alibaba-NLP/gte-Qwen2-1.5B-instruct').get_nearest_neighbors('Can you provide a summary of the latest clinical trials involving KRAS mutations?', n=5, answerable=False)
    print(unansquestion)
        
