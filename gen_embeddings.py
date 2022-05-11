#Importing necessary libraries
import numpy as np
import pandas as pd
import re
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

#Checking for availability of GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#Using pre-trained models for tokenization and embeddings generation
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)
model.eval()

def main():
    
    #Call function to read and Pre-process the data
    res = pre_process()
    print(len(res))
    #To be performed only if GPU in use
    train_y, label_dict = gpu_conversions(res)
    
    #Generate embeddings and write to pkl file
    write_to_pkl_file(train_y, label_dict)
    
#Write generated embeddings to pkl file
def write_to_pkl_file(data, label_dict):
    i = 0
    while i < len(data):
        if i+100 > len(data):
            new_pooled_arr = get_sentence_embeddings(data[i:len(data)], label_dict, i)
        else:
            new_pooled_arr = get_sentence_embeddings(data[i:i+100], label_dict, i)
        i+= 100
        with open('embeddings_stack_overflow_answer_titles.txt', 'ab+') as f:
            pickle.dump(new_pooled_arr, f)
        print(f'{i} embeddings have been created!')
    
#Pre-processing data
def pre_process():
    #Reading stackoverflow data
    data = pd.read_pickle("./formatted_data/unique_answer_titles.pkl")

    #Removing 'nan' values
    data = [x for x in data if str(x) != 'nan']
    
    #Removing white spaces
    res = []
    for item in data:
        if item.strip():
            res.append(item)

    return res

def gpu_conversions(data):
    # dictionary that maps integer to its string value 
    sent_dict = {}

    # list to store integer labels 
    int_sents = []

    for i in range(len(data)):
        sent_dict[i] = data[i]
        int_sents.append(i)
        
    # Now pass this int_labels to the torch.tensor and use it as label.
    train_y = torch.tensor(int_sents)
    
    #convert to cuda tensor
    train_y = train_y.to(device)
    
    return train_y, sent_dict

def get_sentence_embeddings(data, dict_sent, curr_iter):
    tokens = {'input_ids': [], 'attention_mask': []}

    for i in range(0, len(data)):
        sentence = dict_sent[curr_iter + i]
        # tokenize sentence and append to dictionary lists
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                          padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    print("tokens made!")
    
    # Flatten list of tensors into single tensor and convert to cuda tensors if necessary
    tokens['input_ids'] = torch.stack(tokens['input_ids']).to(device)
    tokens['attention_mask'] = torch.stack(tokens['attention_mask']).to(device)
    
    #Pass tokens to model to obtain output
    with torch.no_grad():
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
    
    #Perform mean pooling and convert from cuda tensor to cpu tensor
    mean_pooled = (summed / summed_mask).cpu()
    
    #Convert torch tensor to numpy array
    embeddings_arr = mean_pooled.detach().numpy()

    return embeddings_arr

if __name__ == "__main__":
    main()