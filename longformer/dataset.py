from tqdm import tqdm
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers import AutoTokenizer
model_name = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)


raw_documents_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/crawler/datasets/pdfs/'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/longformer/MLM/ds3/'
min_sentence_length = 3 #In words
max_sentence_length = 100 #In words
min_word_length = 3
max_tokens = 4096


def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        dataset.append(file_path)
    return dataset



raw_ds = loadData(raw_documents_folder,'')
print(len(raw_ds))


train_raw_ds = raw_ds


doc_index = 0
total_chunks_len = 0

dataset = torch.empty(0, dtype=torch.long)

unique_id = 0
documents_length = []
for document in tqdm(train_raw_ds):  

    tokenized_version = 1

    with open(document, 'r') as f:
        content = f.read()
        
        sentences = content.replace('\n', '').replace('\t', '').replace('-OMISSIS-', " ")
        sentences = sentences.split('SENTENZA')
        if(len(sentences) < 2):
            #print(' ASKHDAHSJDHKAHSKJDHAHSJDHAHSDKJHAJSDHJKHASKJDHjk \n\n\n\n\n\n')
            continue
        if(len(sentences) > 2):
            continue
        sentences = sentences[1:]

        # Tokenize the sentence
        tokenized_sent = tokenizer(sentences, return_tensors='pt')['input_ids'].squeeze(0)
        
        # Add the sentence to the dataset, but keep it less than max_tokens
        chunk_len = tokenized_sent.shape[0]
        documents_length.append(chunk_len)
        if(chunk_len > max_tokens):
            tokenized_version = 2
            tokenized_sent_1 = tokenized_sent[:max_tokens]
            tokenized_sent_2 = tokenized_sent[-(max_tokens - 1):]
            cls_token = torch.tensor([101])
            tokenized_sent_2 = torch.cat((cls_token,tokenized_sent_2), dim=0)
            #break
        else:
            tokenized_sent_1 = tokenized_sent
            tokenized_sent_2 = None
        with open(dataset_folder + f'train/train_ds{doc_index}_{unique_id}', 'wb') as f:
            pickle.dump({'version': tokenized_version, 'tokenized_sentence': tokenized_sent_1, 'alt_version': tokenized_sent_2}, f)
        

            
          
    unique_id += 1
    if(unique_id % 1000 == 0):
        print(f'\nMean {np.mean(documents_length)} - Max {np.max(documents_length)} - Min {np.min(documents_length)}')
    
    

print('Saved to')
print(dataset_folder + 'train/train_ds')
      
print('finished!')