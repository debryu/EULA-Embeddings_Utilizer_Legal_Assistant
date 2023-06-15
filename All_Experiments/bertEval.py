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

import wandb


loadModel = True
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds1/'
batch_size = 1
#bert_lr = 0.00000000000001
#mlm_head_lr = 0.000001
bert_lr = 0.00001

use_wandb = False

iterations = 1000
partial_save_every = 10000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

model_name = "bert-base-multilingual-cased"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/saved_models/MLM/bert-base-multilingual-cased/epochs/"

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained(model_name)
vocab_size = len(tokenizer)
print('Vocab tokenizer size', vocab_size)
#tokenized_content = tokenizer.tokenize("my name is earl", return_tensors='pt')
#print(tokenized_content)


wandbConfig = {
 
  "version": "1.0",
  "model used": model_name,
  "batch size": batch_size,
  'learning rate bert': bert_lr,
  "save path": save_path,
}

if(use_wandb):
    wandb.init(project="BERTLAW", entity="nicola-debole", config = wandbConfig)


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
            sentences = data['sentence']
            

            #masked_sentences = data['masked_sentences'] 
            sent_len = data['length']

        return {'sent':sentences,
                #'corr_sent': corr_sentences,
                'sent_len':sent_len,
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
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,  drop_last=True), len(ds)



train_ds = loadData(dataset_folder, 'train/')
dataset_train, dataset_train_length = create_data_loader(train_ds, 1, eval=True)




from transformers import BertForMaskedLM
bertMLM = BertForMaskedLM.from_pretrained(model_name)
bertMLM.to(device)


if(loadModel):
    bertMLM.load_state_dict(torch.load(save_path + 'bertMLM_partial_epoch0.pth', map_location=torch.device(device)))
    print('Model loaded!')

epoch = 1

for i in range(iterations):
    for batch in tqdm(dataset_train):
        #print(batch['sent'])
        #print('\n\n')
        #print(batch['m_sents'])
        #print('\n\n')
        gc.collect()
        torch.cuda.empty_cache()
        tokenized_sentence = batch['sent'].to('cpu')

        
        mask = np.isin(tokenized_sentence[0], 101)
        #print(mask)

        tokenized_sentence = tokenized_sentence.to(device)
        cls_token = torch.tensor([101]).to(device)

        corr_sentences = torch.cat((cls_token,tokenized_sentence[0][~mask]), dim=0).to(device)
        corr_sentences = corr_sentences.unsqueeze(0)

        #print(corr_sentences.shape)
        #print(tokenized_sentence.shape)
        sent_len = batch['sent_len']
        #print(sent_len)


        '''    USE
        The correct tokenized version instead
        '''
        tokenized_sentence = corr_sentences
        '''
        TODO: EDIT THE MASKING PART AND PUT IT IN THE DATASET
        '''

        # create random array of floats in equal dimension to input_ids
        rand = torch.rand(tokenized_sentence.shape).to(device)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < 0.15 * (tokenized_sentence != 101) * (tokenized_sentence != 102)
        unchanged = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.0)
        random_token_mask = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.1)
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, vocab_size, (len(random_token_idx[0]),), device=device)
        final_mask = mask_arr & ~random_token_mask & ~unchanged

        #print(mask_arr.squeeze(0).nonzero().flatten().tolist())
        # create selection from mask_arr
        selection = final_mask.squeeze(0).nonzero().flatten().to(device)
        tokens_to_predict = mask_arr.squeeze(0).nonzero().flatten().to(device)
        #print(selection)
        # apply selection index to inputs.input_ids, adding MASK tokens
        tokenized_masked_sentence = tokenized_sentence.clone().to(device)
        tokenized_masked_sentence[0,selection] = 103
        tokenized_masked_sentence[random_token_idx] = random_tokens

        #print(tokenized_masked_sentence)
        #print(tokenized_sentence)
        #print('\n\n')
        #print('input of the model',tokenized_sentence.shape)
        
        bo = bertMLM(tokenized_sentence).logits.squeeze(0)
        probs = F.softmax(bo, dim=1)
        #print('bert-out:', bo)
        #print(bo.shape)
        #print('bert output', bert_output)
        #print(bert_output.shape)
        
        #print('predictions', tokenizer.convert_ids_to_tokens(pred_tok))
        #print('preds:',prediction.shape)
        #print(prediction.shape)
        #logits = F.softmax(prediction, dim=1)   #ALREADY DOING THAT IN THE LAYER
        
        #print(masked_toks.shape)
        labels = tokenized_sentence.clone().to(device).squeeze(0)[tokens_to_predict].to(device)
        print('True labels\n',tokenizer.convert_ids_to_tokens(labels))
        masked_num = labels.shape[0]
        
        #labels[tokenized_sentence != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens
        #print(masked_toks.shape)
        #print(masked_toks)
        #one_hot_encoding = torch.zeros((masked_num, vocab_size)).scatter_(1, labels.unsqueeze(1), 1).to(device)
        #print(masked_toks.shape)
        #print(one_hot_encoding)

        only_masked_predictions = probs[tokens_to_predict]
        print('Pred labels \n',tokenizer.convert_ids_to_tokens(torch.argmax(only_masked_predictions, dim = 1)))
        loss = F.cross_entropy(only_masked_predictions, labels)
        print(loss, ' \n\n\n')
        #print(masked_toks.shape)
        #print(one_hot_encoding.shape)
        if(use_wandb):
           wandb.log({'loss': loss})
        #print(loss)
        #torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
        
        
        #print(tokenizer.convert_ids_to_tokens(prediction))
        epoch += 1

        if(epoch > 100 and False):
            epoch = 1
            break
