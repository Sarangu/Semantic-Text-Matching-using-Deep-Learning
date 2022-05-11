#This file contains code that involves generating the candidates for an input query

#Importing necessary packages
import numpy as np
import pandas as pd
import re
import torch
import faiss
import os
import pickle
import sys
from transformers import AutoTokenizer, AutoModel
from gen_embeddings import pre_process 

#Using pre-trained models for tokenization and embeddings generation
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

def get_sentence_embeddings(data):
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in data:
        # tokenize sentence and append to dictionary lists
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                          padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    # Flatten list of tensors into single tensor and convert to cuda tensors if necessary
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    #Pass tokens to model to obtain output
    outputs = model(**tokens)
    
    #Obtain embeddings as last hidden state
    embeddings = outputs.last_hidden_state

    #Create mask
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    #Compute masked embeddings
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    #Perform mean pooling
    mean_pooled = (summed / summed_mask)
    
    #Convert torch tensor to numpy array
    embeddings_arr = mean_pooled.detach().numpy()

    return embeddings_arr

def obtain_trained_embeddings():
    #Reading embeddings from pkl file into a list
    objects = []
    with (open("./embeddings_stack_overflow_answer_titles.txt", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    embeddings = [item for sublist in objects for item in sublist]
    return embeddings

def faiss_index(embeddings):
    #Length of each SBERT embedding is 768
    d= 768
    
    #Creating faiss indices, adding for all embeddings
    index = faiss.IndexFlatL2(d)
    index.add(np.stack(embeddings, axis=0))
    
    return index

def generate_candidates(query):
    query = [sys.argv[1]]
    query_embedding = get_sentence_embeddings(query)
    
    #Obtain pre-processed data
    res = pre_process()
    
    #Calling functions to obtain embeddings of corpus and create faiss index
    embeddings = obtain_trained_embeddings()
    print(len(embeddings))
    index = faiss_index(embeddings)
    
    candidates = []
    for query, query_embedding in zip(query, query_embedding):
        distances, indices = index.search(np.asarray(query_embedding).reshape(1,768),20)
        
        for idx in range(0,20):
            candidates.append(res[indices[0,idx]])
    return candidates

def main():
    if len(sys.argv) < 2:
        print(f"Incorrect Usage, kindly enter your query")
        return
    
    query = [sys.argv[1]]
    final_candidates = generate_candidates(query)
if __name__ == "__main__":
    main()