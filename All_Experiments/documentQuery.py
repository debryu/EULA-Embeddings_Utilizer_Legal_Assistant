from transformers import pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
import os
import torch.nn.functional as F
from tqdm import tqdm 
import gc
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

model_name = "bert-base-multilingual-cased"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/saved_models/MLM/bert-base-multilingual-cased/epochs/"

from transformers import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(model_name)
bertMLM = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
bertMLM.to(device)

bertMLM.load_state_dict(torch.load(save_path + 'bertMLM_partial_epoch0.pth', map_location=torch.device(device)))
print('Model loaded!')

def getEmbedding(text):
    tokenized_text = torch.tensor(tokenizer.encode(text, return_tensors='pt').to(device))
    output = bertMLM(tokenized_text)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token

def getEmbeddingFromTokens(tokenized_text):
    output = bertMLM(tokenized_text)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token


import os
import pinecone

# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY') or '7c4fa8c6-f08d-4b28-8797-3287e1a42cf0'
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT') or 'asia-southeast1-gcp'

print('init pinecone')

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

print('pinecone init done')

index_name = 'semantic-search-bertlaw'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,
        metric='cosine'
    )

# now connect to the index
index = pinecone.GRPCIndex(index_name)

query_embs = getEmbedding('studente denuncia per abuso d ufficio professore universitario').to('cpu').tolist()

query_response = index.query(
    namespace='',
    top_k=2,
    include_values=False,
    include_metadata=True,
    vector = query_embs,
    
)

print('\n\n\n\n',query_response)
#index.delete(ids=['1', 'vec2'], namespace='example-namespace')


