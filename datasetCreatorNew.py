from tqdm import tqdm
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers import BertTokenizer
model_name = "allenai/longformer-base-4096"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_documents_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/crawler/datasets/raw_pdfs/'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/longformer/MLM/ds_selected/'
vocab_size = len(tokenizer)

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
unique_id = 0
for document in tqdm(train_raw_ds):  
    
    with open(document, 'rb') as f:
        data = pickle.load(f)
        text = data['text']
        id = data['counter']
        
        # Basic preprocessing
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        text = text.split('SENTENZA')
        if(len(text) < 2):
            continue
        text = text[1]


        # Select the text
        text = text.split('FATTO')
        if(len(text) < 2):
            continue
        text = text[1]
        
        # Tokenize the text
        tokenized_text = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').squeeze(0)
        #print(tokenized_text.shape)

        ground_truth = tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length = 4096, truncation=True, return_tensors='pt').squeeze(0)
        #print(ground_truth.shape)
        
        # Create a chunk of random size
        chunk_size = np.random.randint(40, 4090)

        # Extract the chunk from the text
        chunk = tokenized_text[:chunk_size]
        
        # Add special tokens
        chunk = torch.cat([torch.tensor([tokenizer.cls_token_id]), chunk, torch.tensor([tokenizer.sep_token_id])])
        #print(chunk)
        #print(tokenizer.decode(chunk))


        # Pad the chunk
        chunk = nn.functional.pad(chunk, (0, 4096 - chunk_size - 2), value=tokenizer.pad_token_id)
        #print(tokenizer.decode(chunk))

        # Mast tokens in the chunk
        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(chunk.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < 0.15 * (chunk != tokenizer.cls_token_id) * (chunk != tokenizer.sep_token_id) * (chunk != tokenizer.pad_token_id)
        unchanged = mask_arr & (torch.rand(mask_arr.shape) < 0.3)
        random_token_mask = mask_arr & (torch.rand(mask_arr.shape) < 0.3)
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, vocab_size, (len(random_token_idx[0]),))
        final_mask = mask_arr & ~random_token_mask & ~unchanged



        # create selection from mask_arr
        selection = final_mask.squeeze(0).nonzero().flatten()
        #print(selection)
        tokens_to_predict = mask_arr.squeeze(0).nonzero().flatten()
        #print(selection)
        #print(tokenized_sentence.shape)
        # apply selection index to inputs.input_ids, adding MASK tokens
        tokenized_chunk = chunk.clone()
        tokenized_chunk[selection] = tokenizer.mask_token_id
        tokenized_chunk[random_token_idx] = random_tokens

        #print(tokenized_chunk.shape)
        #print(tokenized_chunk)
        #print(chunk.shape)
        #print(chunk)
        #print(ground_truth.shape)
        #print(ground_truth)

        with open(dataset_folder + f'train_{id}', 'wb') as f:
            pickle.dump({'groundT': ground_truth, 'masked_chunk': tokenized_chunk, 'original_chunk': chunk}, f)
        
        