from rank_bm25 import BM25Okapi
import pickle
import numpy as np
current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/longformer/'

with open(current_folder + 'data/LFtrainedOnHFdataset.pickle', 'rb') as file:
    data = pickle.load(file)

embeddings = np.array(data['embeddings'])
texts = data['raw']
ids = np.array(data['id'])

#print(texts[0].replace('<pad>', '').replace('<s>', '').replace('</s>', ''))

corpus = [text.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for text in texts]

tokenized_corpus = [doc.split(" ") for doc in corpus]
print(len(tokenized_corpus))
bm25 = BM25Okapi(tokenized_corpus)

# Test number 1
query = "diniego del visto d'ingresso per motivi di turismo"
# Test number 2
query = 'Non mi è stato riconosciuto il titolo di laurea magistrale conseguito in Germania. Vorrei presentare ricorso.'

tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print('doc_scores: ', doc_scores)

results = bm25.get_top_n(tokenized_query, corpus, n=10)

print(results[0])
print('\n\n')
print(results[1])
print('\n\n')
print(results[2])
print('\n\n')
print(results[3])
print('\n\n')
print(results[4])
print('\n\n')
print(results[5])
print('\n\n')
print(results[6])
print('\n\n')
print(results[7])
print('\n\n')
print(results[8])
print('\n\n')
print(results[9])