from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
dataset = load_dataset("joelito/mc4_legal", "it", split='train', streaming=True)
dataset_save_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds_newds2/'
vocab_size = len(tokenizer)

#CREATE THE DATALOADER
def create_data_loader(data, batch_size, eval=False):
    ds = data
    if not eval:
        return DataLoader(ds, batch_size=batch_size, drop_last=True)

    else:
        return DataLoader(ds, batch_size=batch_size,  drop_last=True)


dataset_train = create_data_loader(dataset, 1, eval=False)



unique_id = 0

for batch in tqdm(dataset_train):
    text = batch['text'][0]
    
    # Split the text in smaller chunks 
    
    #print(text)
    #print(len(text))

    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        
        # Skip if the paragraph is too short
        if(len(paragraph) < 200):
            continue

        ground_truth = tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length = 4096, truncation=True)
        raw_tok = tokenizer.encode(text,add_special_tokens=False, max_length = 4096, truncation=True)
        #print(raw_tok)
        
        #print(ground_truth)
        attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in ground_truth]
        #print(attention_mask)
        #print('\n')
        
        # Count the tokens
        n_toks = len(raw_tok)
        target_n_of_masked_toks = 150
        probability = target_n_of_masked_toks / n_toks

        # Mast tokens in the chunk
        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(len(raw_tok)-5)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < probability
        unchanged = mask_arr & (torch.rand(mask_arr.shape) < 0.3)
        random_token_mask = mask_arr & (torch.rand(mask_arr.shape) < 0.3)
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, vocab_size, (len(random_token_idx[0]),))
        final_mask = mask_arr & ~random_token_mask & ~unchanged



        # create selection from mask_arr
        selection = final_mask.squeeze(0).nonzero().flatten() + 1
        
        
        tokens_to_predict = mask_arr.squeeze(0).nonzero().flatten() + 1
        #print(tokens_to_predict)
        #print(selection)
        #print(tokenized_sentence.shape)
        # apply selection index to inputs.input_ids, adding MASK tokens
        ground_truth = torch.tensor(ground_truth)
        attention_mask = torch.tensor(attention_mask)
        masked_chunk = ground_truth.clone()
        masked_chunk[selection] = tokenizer.mask_token_id
        masked_chunk[random_token_idx] = random_tokens


        if(ground_truth.shape[0] != 4096 or attention_mask.shape[0] != 4096 or masked_chunk.shape[0] != 4096):
            continue
        #print(ground_truth.shape[0])
        #print(attention_mask.shape[0])
        #print(masked_chunk.shape[0])
        #print('\n')

        with open(dataset_save_folder + f'train_{unique_id}', 'wb') as f:
            pickle.dump({'groundT': ground_truth, 
                         'masked_chunk': masked_chunk, 
                         'attention_mask': attention_mask, 
                         'unique_id': unique_id,
                         'list_of_predictions' : tokens_to_predict
                         }
                         , f)

        unique_id += 1

    #break


