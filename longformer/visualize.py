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
checkpoint = 'longformerNewDifferentDataset_partial_epoch88886.pth'
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/saved_models/MLM/longformer-base-4096/epochs/"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = len(tokenizer)
print('Vocab tokenizer size', vocab_size)
#tokenized_content = tokenizer.tokenize("my name is earl", return_tensors='pt')
#print(tokenized_content)

from transformers import AutoModel, AutoModelForMaskedLM
longformerMLM = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
longformerMLM.to(device)


longformerMLM.load_state_dict(torch.load(save_path + checkpoint, map_location=torch.device(device)))
print('Model loaded!')

longformerMLM.eval()

#def getEmbedding(text):
#    tokenized_text = torch.tensor(tokenizer.encode(text, return_tensors='pt').to(device))
#    output = longformerMLM(tokenized_text)
#    last_layer = output['hidden_states'][-1].squeeze(0)
#    cls_token = last_layer[0].squeeze(0)
#    return cls_token

def getEmbedding(text):
    ground_truth = tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length = 4096, truncation=True)
    #print(ground_truth)
    attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in ground_truth]
    
    input_text = torch.tensor(ground_truth).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

    #print(ground_truth)
    output = longformerMLM(input_text, attention_mask = attention_mask)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token

def getEmbeddingFromTokens(tokenized_text):
    output = longformerMLM(tokenized_text)
    #print(output)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token


index_name = 'semantic-search-alex'


from tqdm.auto import tqdm

batch_size = 128

current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/longformer/'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds_selected/'

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
            #pickle.dump({'groundT': ground_truth, 'masked_chunk': tokenized_chunk, 'original_chunk': chunk}, f)
            groundT = data['groundT']
            #print(groundT.shape)
            masked_chunk = data['masked_chunk']
            if(masked_chunk.shape[0] != 4096):
                #add padding
                pad = torch.ones( 4096 - masked_chunk.shape[0], dtype=torch.long)
                masked_chunk = torch.cat((masked_chunk, pad), dim=0)

            
            original_chunk = data['original_chunk']
            if(original_chunk.shape[0] != 4096):
                #add padding
                pad = torch.ones( 4096 - original_chunk.shape[0], dtype=torch.long)
                original_chunk = torch.cat((original_chunk, pad), dim=0)
            #print(original_chunk.shape)

        return {'gt': groundT,
                'm_chunk': masked_chunk,
                'o_chunk': original_chunk
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

train_ds = loadData(dataset_folder,'')
dataset_train, dataset_train_length = create_data_loader(train_ds, 1, eval=True)

ids = 0
embeddings = []
raw_texts = []
indexes = []
for batch in tqdm(dataset_train):
    # find end of batch
   
    # create IDs batch
    ids += 1
    # create metadata batch
    text = batch['o_chunk'].to(device)
    
    raw_text = tokenizer.decode(text[0])
    #print(raw_text[0:100])
    #print(text)
    # create embeddings
    embedding = getEmbeddingFromTokens(text).to('cpu').tolist()
    #print(embedding.shape)
    # create records list for upsert
    raw_texts.append(raw_text)
    indexes.append(ids)
    embeddings.append(embedding)
    # upsert to Pinecone
    #print(embedding)
    
    # check number of records in the index
    if ids % 2000 == 0:
        with open(current_folder + 'data/LF88886.pickle', 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'id': indexes, 'raw': raw_texts}, f)
        break

