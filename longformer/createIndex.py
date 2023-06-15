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

from tqdm.auto import tqdm

batch_size = 128


dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds3/'

class MLMdataset(Dataset):

    def __init__(self, data):
        
        #load the data
        self.pickleFiles = data

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.pickleFiles)


    def __getitem__(self, idx):
        
        with open(self.pickleFiles[idx], 'rb') as file:             
            data = pickle.load(file)
            versions = data['version']
            doc = data['tokenized_sentence']

        return {'sent':doc,
                #'corr_sent': corr_sentences,
                'ver':versions,
                #'m_sents':masked_sentences,
              }
    
def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        dataset.append(file_path)
    return dataset

#CREATE THE DATALOADER
def create_data_loader(data, batch_size, eval=False):
    ds = MLMdataset(data)
    if not eval:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,  drop_last=True), len(ds)

train_ds = loadData(dataset_folder,'train')
dataset_train, dataset_train_length = create_data_loader(train_ds, 1, eval=True)

ids = 0
for batch in tqdm(dataset_train):
    # find end of batch
   
    # create IDs batch
    ids += 1
    # create metadata batch
    text = batch['sent'].to(device)
    

    #print(text)
    # create embeddings
    embedding = getEmbeddingFromTokens(text).to('cpu').tolist()
    #print(embedding.shape)
    # create records list for upsert
    name = f'vector{str(ids)}'
    
    text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(text[0]))
    metadata = {'raw_text': text}
    # upsert to Pinecone
    #print(embedding)
    index.upsert([(name, embedding, metadata)])
    
    if(ids > 2000):
        break
# check number of records in the index
index.describe_index_stats()



query_embs = getEmbedding('studente denuncia per abuso dufficio professore universitario').to('cpu').tolist()

query_response = index.query(
    namespace='',
    top_k=3,
    include_values=False,
    include_metadata=True,
    vector = query_embs,
    
)
print(query_response)
#index.delete(ids=['1', 'vec2'], namespace='example-namespace')


