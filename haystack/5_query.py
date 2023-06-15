import os
from tqdm import tqdm
import pickle
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# PARAMETERS
DOCUMENTS_TO_RETRIEVE = 5

# WRITE YOUR QUERY HERE
query = "I ricorrenti hanno partecipato alla selezione per l’ammissione al corso di laurea in Ingegneria. Questi, pur risultando idonei, non sono stati immatricolati all’indicato corso di laurea."

# DATASETS
dataset_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_Crawler/datasets/raw_pdfs/"
documentStore_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/index/"
bm25_index = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/raw_for_bm25/10000_INDEX.pickle"

# LOAD THE MODEL
model = SentenceTransformer("efederici/sentence-BERTino")

# Compute the embedding of the query
query_embedding = model.encode(query)

# Define the similarity measure
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# load all the paths of the documents
def loadData(path):
    files =  os.listdir(path)
    dataset = []
    for file in files:
        file_path = os.path.join(path, file)
        dataset.append(file_path)
    return dataset

# Load the documents (they were csv files, now they are pickle files)
# So this line is useless
embeddings_csv_files = loadData(documentStore_folder)
# Load all the documents for later (at the end where we show the results)
sentenze_files = loadData(dataset_folder)

with open(bm25_index, 'rb') as f:
    bm25_index_loaded = pickle.load(f)
bm25 = BM25Okapi(bm25_index_loaded['tok_docs'])

tokenized_query = query.split(" ")
# Retrieve the documents  
retrieved = bm25.get_top_n(tokenized_query, bm25_index_loaded['docs'], n=DOCUMENTS_TO_RETRIEVE)

# Split by '-ID%:' and take the second part, I added the id in the text of the document so that I can retrieve it here
ids_retrieved = [retrieved[i].split('-ID%:')[1] for i in range(len(retrieved))]
print(ids_retrieved)

'''
COMPUTE THE EMBEDDINGS ONLY ON THE RETRIEVED DOCUMENTS AND FIND THE MOST RELEVANT ONE SEMANTICALLY
'''
results = []
for i,ids_retrieved in enumerate(ids_retrieved):
    embedding_file_path = documentStore_folder + f'index_{ids_retrieved}.pickle'
    # Check if the file exists
    if not os.path.exists(embedding_file_path):
        print("File does not exist (it has not been preprocess). Forget about it.")
        continue
    
    with open(embedding_file_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    similarities = [cosine_similarity(query_embedding, np.array(embedding)) for embedding in embeddings]     
    results.append({'max': max(similarities),'argmax': np.argmax(similarities), 'mean': np.mean(similarities), 'min': min(similarities), 'name_id': ids_retrieved})

'''
------------------------------------------------------------------------------------------
THIS BLOCK IS FOR QUERYING ALL THE DOCUMENTS IN A DENSE WAY
'''
'''
results = []

for i,document in enumerate(sentenze_files):
    with open(document, 'rb') as f:
        data = pickle.load(f)
    name_id = data['id']
    content = data['text']
    counter = data['counter']
    # REPLACE \n WITH WHITESPACE
    content = content.replace('\t', '   ').replace('\n', '   ').replace('\r', '   ')
    location = data['location']
    link = data['link']
    #print(name_id)
    embedding_file_path = documentStore_folder + f'index_{name_id}.pickle'

    # Check if the file exists
    if not os.path.exists(embedding_file_path):
        print("File does not exist.")
        break
    

  
    with open(embedding_file_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    

    similarities = [cosine_similarity(query_embedding, np.array(embedding)) for embedding in embeddings]     


    results.append({'max': max(similarities),'argmax': np.argmax(similarities), 'mean': np.mean(similarities), 'min': min(similarities), 'name_id': name_id, 'location': location, 'link': link, 'counter': counter})
    

    #if i == 5000:
    #    break
'''


# Choose how to sort the results
sorted_dict = sorted(results, key=lambda x: x['max'], reverse=True)
print(sorted_dict[0]['max']) 
print(sorted_dict[0]['mean'])


print('\n- STATS: -\n')
for i,item in enumerate(sorted_dict):
    print(item['max'],item['argmax'],item['mean'],item['min'])
    if i == 10:
        break


# Deliver the document with the highest similarity
ID_TO_LOOK_FOR = 0
while ID_TO_LOOK_FOR < len(sorted_dict):
    print('Searching...')
    for i,document in enumerate(sentenze_files):
        with open(document, 'rb') as f:
            data = pickle.load(f)
        name_id = data['id']
        if(name_id == sorted_dict[ID_TO_LOOK_FOR]['name_id']):
            content = data['text']
            counter = data['counter']
            # REPLACE \n WITH WHITESPACE
            content = content.replace('\t', '   ').replace('\n', '   ').replace('\r', '   ')
            location = data['location'].replace('(', '').replace(',', '')
            link = data['link']
            print(f'\n\nHere is what I found for you: \n- Link:  {link} \n- Location: {location} \n- Rank: {ID_TO_LOOK_FOR+1}')
            break
        else:
            continue
        
    sec = input('Press ENTER to find another one...\n')
    ID_TO_LOOK_FOR += 1


print('You viewed all results! Terminating...')


'''
------------------------------------------------------------------------------------------
CODE TO CHECK WHICH SENTENCE IS THE ONE WITH THE HIGHEST SIMILARITY IN THE DOCUMENT

'''


'''
c = sorted_dict[0]['counter']
uid = sorted_dict[0]['name_id']
argmax = sorted_dict[0]['argmax']


from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
converter = TextConverter(remove_numeric_tables=True, valid_languages=["it"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=1,
    split_respect_sentence_boundary=False,
)

doc_txt = converter.convert(file_path=dataset_folder + f'document_{c}_{uid}.pickle', meta=None)[0]
docs_default = preprocessor.process([doc_txt])

print(docs_default[argmax].content)

'''