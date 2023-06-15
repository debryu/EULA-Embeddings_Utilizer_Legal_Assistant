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

virtualBatch = 1
loadModel = False
continue_from_epoch = 0
model_to_load = 'longformerNew_partial_epoch1.pth'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds_newds2/'
batch_size = 1
#bert_lr = 0.00000000000001
mlm_head_lr = 0.00001
lf_lr = 0.000005

use_wandb = True

iterations = 1000
partial_save_every = 4000
#stop_after = 10000
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(10)
num_workers = 6 #6
print(device)

model_name = "allenai/longformer-base-4096"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/saved_models/MLM/longformer-base-4096/epochs/"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = len(tokenizer)
print('Vocab tokenizer size', vocab_size)
#tokenized_content = tokenizer.tokenize("my name is earl", return_tensors='pt')
#print(tokenized_content)



wandbConfig = {
 
  "version": "1.0",
  "model used": model_name,
  "batch size": batch_size,
  "grad accumulation batch": virtualBatch,
  'learning rate longformer': lf_lr,
  "save path": save_path,
}





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
            
            attention_m = data['attention_mask']
            unique_id = data['unique_id']
            preds = data['list_of_predictions']

        return {'gt': groundT,
                'm_chunk': masked_chunk,
                'att_mask': attention_m,
                'uid': unique_id,
                'preds': preds
              }


def main():

    if(use_wandb):
        wandb.init(project="BERTLAW", entity="nicola-debole", config = wandbConfig)

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
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers = num_workers, drop_last=True), len(ds)

        else:
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers = num_workers, drop_last=True), len(ds)



    train_ds = loadData(dataset_folder, '')
    dataset_train, dataset_train_length = create_data_loader(train_ds, batch_size, eval=True)




    from transformers import AutoModel, AutoModelForMaskedLM
    longformerMLM = AutoModelForMaskedLM.from_pretrained(model_name)
    longformerMLM.to(device)
    longformerMLM.train()



    if(loadModel):
        longformerMLM.load_state_dict(torch.load(save_path + model_to_load, map_location=torch.device(device)))
        print('Model loaded!')

    par1 = longformerMLM.parameters()


    optim = torch.optim.AdamW([
        {'params': par1, 'lr': lf_lr},
        
        ])
        
    epoch = 1

    for i in range(iterations):
        for batch in tqdm(dataset_train):
            #print(batch['sent'])
            #print('\n\n')
            #print(batch['m_sents'])
            #print('\n\n')
            #print(epoch)
            if(epoch < continue_from_epoch):
                epoch += 1
                continue
            gc.collect()
            torch.cuda.empty_cache()
            masked_chunks = batch['m_chunk'].to(device)
            gt = batch['gt'].to(device).squeeze(0)
            att_mask = batch['att_mask'].to(device)
            pred_indexes = batch['preds'].to(device)

            out = longformerMLM(masked_chunks, attention_mask = att_mask).logits.squeeze(0)

            only_pred_tokens = out[pred_indexes].squeeze(0)
            only_pred_gt = gt[pred_indexes].squeeze(0)
            #print('-> in:', masked_chunks.shape)
            #print('-> out:', out.shape)
            #print('-> gt:', gt.shape)
            prediction = torch.argmax(out, dim=1)
            tokens_predicted = tokenizer.convert_ids_to_tokens(prediction)
            #loss = F.cross_entropy(out, gt)
            #print(only_pred_tokens.shape)
            #print(only_pred_gt.shape)
            loss = F.cross_entropy(only_pred_tokens, only_pred_gt)
            #print(loss)
            #print(masked_toks.shape)
            #print(one_hot_encoding.shape)
            if(use_wandb):
                wandb.log({'loss': loss})
                wandb.log({'n_predictions': only_pred_gt.shape[0]})
            #print(loss)

            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            loss = loss / virtualBatch
            loss.backward() 
            
            
            #torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
            
            if(epoch % virtualBatch == 0):
                #print('OPTIMIZING')
                optim.step() 
                optim.zero_grad()
            
            #print(tokenizer.convert_ids_to_tokens(prediction))
            if(epoch % partial_save_every == 0):
                torch.save(longformerMLM.state_dict(), save_path + f'longformerNewDifferentDataset_partial_epoch.pth')
            epoch += 1

            #if(epoch > stop_after):
            #    break


        
        torch.save(longformerMLM.state_dict(), save_path + f'longformerNew_epoch{i}.pth')


if __name__ == "__main__":
    main()