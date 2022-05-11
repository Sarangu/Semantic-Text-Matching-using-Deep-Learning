# COMSW4995 Semantic Representations for NLP - Semantic-Text-Matching-using-Deep-Learning

## Team Members
Aishwarya Sarangu (als2389)

Anusha Holla (ah3816)

David Michael Smythe (dms2313)

Priyanka Mishra (pm3105)

Xiaohan Feng (xf2198)

## Submitted Files
* **generate_embeddings.py:** script to generate sentence embeddings for titles of all answers in the corpus
* **candidate_generation.py:** script to run to generate top 'k' candidates (titles) for a user input query
* **information_retrieval.py:** script to rerank and generate final answers for user input query
* **transcript.txt:** contains sample run of ```information_retrieval```
* **embeddings_stack_overflow_answer_titles.txt:** Download file using drive link in root directory - contains embeddings generated for 294,444 titles after runnning ```generate_embeddings```


 Link to download ```embeddings_stack_overflow_answer_titles``` : https://drive.google.com/file/d/14HhD0uQCo7gZOWE9SzZoqcv5MTE3jjIU/view?usp=sharing
 
 
## Installing Dependencies

```
$ sudo apt-get update
$ pip install numpy
$ pip install pandas
$ pip install torch torchvision torchaudio
$ pip install transformers
$ pip install faiss-cpu
```

## Running the program

```
$ ./information_retrieval.py "<search_query>"
```

## Project Design
  Despite having multiple state-of-the-art information retrieval methods, one challenge we often face is the time/efficiency of the retrieval system. Since answers are typically much longer in comparison to the titles, computing the embeddings for them and then comparing each of them to the input query would be computationally inefficient and long. This project aims at creating a more efficient retrieval system by generating a small set of candidates based on answer titles, and then comparing the answers of all these candidates with the query. In order to generate top 'k' candidates, we use FAISS index, since this is much faster than other methods such as cosine similarity. For the reranking step which then has a much smaller corpus (20 candidates instead of 30 million), we rank them based on cosine similarity. 
  
We generate sentence embeddings by performing a max-pooling at the last hidden layer of embeddings. This helps in retaining the contextual/semanntic sense of the sentence, thus performing better for sentence-comparison tasks. The project has 3 core files, each of which serve a distinct purpose:
  
  The file ```generate_embeddings```, has the following functionality:
 * **pre-process:** This function reads in unique answer titles from ```unique_answer_titles.pkl```, removes 'nan' values and strips off white spaces.
 * **get_sentence_embeddings:** This function contains the core logic behind creating sentence embeddings for the unique answer titles. It performs the following steps:
    * Each sentence is encoded into input_ids and attention masks
    * Tokens are passed to a pre-trained BERT model and last hidden state of this output is consdiered for embeddings
    * Compute masked embeddings and perform mean pooling
This process is performed in batches of 100 sentences and after each batch of embeddings is generated, it is appended to the final embeddings file: ```embeddings_stack_overflow_answer_titles.txt```.


The file ```candidate_generation``` has the following functionality:
* After obtaining the input query from the user, it first computes the sentence embedding for it using ```.get_sentence_embeddings``` from ```generate_embeddings```.
* Using the embeddings file (for titles), it creates an FAISS index on them
* The top 'k' titles most similar to input query based on FAISS metric are sent to the reranking step.


The file ```information_retrieval``` has the following functionality:
* Obtain top 'k' titles from ```candidate_generation``` and map them to the corresponding answers in the corpus.
* Compute sentence embeddings of these answers
* Finally rank the answers based on cosine similarity between them and user input query, and return to the user

<img src="/Assets/Poster.jpg" alt="Alt text" title="Flow diagram">
