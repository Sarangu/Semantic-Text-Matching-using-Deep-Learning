import numpy as np
import pandas as pd
import re
import torch
import faiss
import time
import os
import pickle
import sys
from scipy import spatial

from transformers import AutoTokenizer, AutoModel
from candidate_generation import get_sentence_embeddings,generate_candidates,faiss_index
qa_pairs = pd.read_pickle('../formatted_data/stackoverflow/answer_title_body_lookup.pkl')
qa_pairs_dict = dict(qa_pairs)

def reranking(title_list,query):
    
    answers_list = []
    query_embedding = get_sentence_embeddings(query)
    
    for idx in range(0,10):
        answers_list.append(qa_pairs_dict[title_list[idx]])
        #print(idx+1,":",qa_pairs_dict[question_list[idx]])
    answer_embeddings = get_sentence_embeddings(answers_list)
    
    scored_answers = {}
    for idx,answer_embedding in enumerate(answer_embeddings):
        result = 1 - spatial.distance.cosine(query_embedding, answer_embedding)
        scored_answers[result]=answers_list[idx]
    ranked_answers = dict(sorted(scored_answers.items(), key=lambda item: item[0],reverse = True))
    
    return ranked_answers

    
    


def main():
    if len(sys.argv) < 2:
        print(f"Incorrect Usage, kindly enter your query")
        return
    
    query = [sys.argv[1]]
    
    question_list = generate_candidates(query)
    print("Candidates generated...Reranking")
            
    ranked_answers = reranking(question_list,query)
    
    print("==========================")
    print("Retrieved Descriptions:")
    print("==========================") 
    for idx,ranked_answer in enumerate(ranked_answers.values()):
        print(idx+1,":",ranked_answer)
        print()
            
if __name__ == "__main__":
    main()