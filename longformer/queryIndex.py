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

model_name = "allenai/longformer-base-4096"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/saved_models/MLM/longformer-base-4096/epochs/"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = len(tokenizer)
print('Vocab tokenizer size', vocab_size)
#tokenized_content = tokenizer.tokenize("my name is earl", return_tensors='pt')
#print(tokenized_content)

from transformers import AutoModel, AutoModelForMaskedLM
longformerMLM = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
longformerMLM.to(device)


longformerMLM.load_state_dict(torch.load(save_path + 'longformer_partial_epoch8000.pth', map_location=torch.device(device)))
print('Model loaded!')

longformerMLM.eval()

def getEmbedding(text):
    tokenized_text = torch.tensor(tokenizer.encode(text, return_tensors='pt').to(device))
    output = longformerMLM(tokenized_text)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token

def getEmbeddingFromTokens(tokenized_text):
    output = longformerMLM(tokenized_text)
    #print(output)
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

index_name = 'semantic-search-alex'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,
        metric='cosine'
    )

# now connect to the index
index = pinecone.GRPCIndex(index_name)

#query_embs = getEmbedding('Sono andato a scuola e ho iniziato ad offrire merendine e bibite ma mi è stato comunicato dalla polizia di non poter continuare.').to('cpu').tolist()
#studente denuncia per abuso dufficio professore universitario
query_embs = getEmbedding('Ottenere il riconoscimento del titolo di studio ottenuto allestero (in Germania)').to('cpu').tolist()
query_response = index.query(
    namespace='',
    top_k=2,
    include_values=False,
    include_metadata=True,
    vector = query_embs,
    
)

print('\n\n\n\n',query_response)
#index.delete(ids=['1', 'vec2'], namespace='example-namespace')


