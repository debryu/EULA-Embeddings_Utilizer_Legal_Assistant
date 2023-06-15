from tqdm import tqdm
import os
import torch
import torch.nn as nn
import pickle
import numpy as np
from transformers import BertTokenizer
model_name = "bert-base-multilingual-cased"
tokenizer1 = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

raw_documents_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/crawler/datasets/pdfs/'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/longformer/MLM/ds3/'
min_sentence_length = 3 #In words
max_sentence_length = 100 #In words
min_word_length = 3
max_tokens = 512


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
for document in tqdm(train_raw_ds):  
    
    with open(document, 'r') as f:
        content = f.read()
        
        

        #print(content)
        #print(document)
        # Split content into paragraphs
        sentences = content.split('\n')

        # Remove all other useless stuff
        sentences = [sentence.replace('\t', '') for sentence in sentences]
        #print(tokenizer.convert_ids_to_tokens(tokenized_content))

        # For each sentence, tokenize it
        for j,sentence in enumerate(sentences):
            
            words = sentence.split(' ')
            # First, leave out the sentence if it is too short or too long
            if(len(words) < min_sentence_length or len(words) > max_sentence_length):
                continue

            # Tokenize the sentence
            tokenized_sent = tokenizer1(' '.join(words), return_tensors='pt')['input_ids'].squeeze(0)
            
            
            # Add the sentence to the dataset, but keep it less than max_tokens
            chunk_len = tokenized_sent.shape[0]
            

            total_chunks_len += chunk_len 
            #print(total_chunks_len)
            if(total_chunks_len > max_tokens):
                total_chunks_len -= chunk_len
                
                ''' 
            
            
                '''

                tokenized_sentence = dataset.unsqueeze(0)
                print(tokenized_sentence)
            
                mask = np.isin(tokenized_sentence[0], 101)
                #print(mask)

                tokenized_sentence = tokenized_sentence
                cls_token = torch.tensor([101])

                corr_sentences = torch.cat((cls_token,tokenized_sentence[0][~mask]), dim=0)
                corr_sentences = corr_sentences.unsqueeze(0)

                tokenized_sentence = corr_sentences
                dataset = tokenized_sentence
                '''
                
                
                '''
                # Save the dataset
                with open(dataset_folder + f'train/train_ds{doc_index}_{unique_id}', 'wb') as f:
                    pickle.dump({'tokenized_sentence': dataset, 'length': chunk_len}, f)
                #print(dataset)
                unique_id += 1
                total_chunks_len = 0
                # Clear the dataset tensor
                dataset = torch.empty(0, dtype=torch.long)

            else:
                dataset = torch.cat((dataset, tokenized_sent), 0)
                

    if(total_chunks_len > 0):

        ''' 
            
            
        '''

        tokenized_sentence = dataset.unsqueeze(0)
        print(tokenized_sentence)
    
        mask = np.isin(tokenized_sentence[0], 101)
        #print(mask)

        tokenized_sentence = tokenized_sentence
        cls_token = torch.tensor([101])

        corr_sentences = torch.cat((cls_token,tokenized_sentence[0][~mask]), dim=0)
        corr_sentences = corr_sentences.unsqueeze(0)

        tokenized_sentence = corr_sentences
        dataset = tokenized_sentence
        '''
        
        
        '''
        # Save the dataset
        with open(dataset_folder + f'train/train_ds{doc_index}_{unique_id}', 'wb') as f:
            pickle.dump({'tokenized_sentence': dataset, 'length': chunk_len}, f)
        #print(dataset)
        
        # Clear the dataset tensor
        dataset = torch.empty(0, dtype=torch.long)
        
    total_chunks_len = 0       
    unique_id = 0
    doc_index += 1
    
    

print('Saved to')
print(dataset_folder + 'train/train_ds')
      
print('finished!')