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

virtualBatch = 8
loadModel = True
continue_from_epoch = 24000
model_to_load = 'longformer_partial_epochIDK4.pth'
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds3/'
batch_size = 1
#bert_lr = 0.00000000000001
mlm_head_lr = 0.00001
lf_lr = 0.0001

use_wandb = True

iterations = 1000
partial_save_every = 4000
#stop_after = 10000
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(8)
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
            versions = data['version']
            doc = data['tokenized_sentence']

        return {'sent':doc,
                #'corr_sent': corr_sentences,
                'ver':versions,
                #'m_sents':masked_sentences,
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
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers = 6, drop_last=True), len(ds)

        else:
            return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers = 6, drop_last=True), len(ds)



    train_ds = loadData(dataset_folder, 'train/')
    dataset_train, dataset_train_length = create_data_loader(train_ds, 1, eval=True)




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
            tokenized_sentence = batch['sent'].to(device).squeeze(0)
            #print(tokenized_sentence)
            #toks = tokenizer.convert_ids_to_tokens(tokenized_sentence)
            #print(tokenizer.convert_tokens_to_string(toks))

            # create random array of floats in equal dimension to input_ids
            rand = torch.rand(tokenized_sentence.shape).to(device)
            # where the random array is less than 0.15, we set true
            mask_arr = rand < 0.15 * (tokenized_sentence != tokenizer.cls_token_id) * (tokenized_sentence != tokenizer.sep_token_id)
            unchanged = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.15)
            random_token_mask = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.6)
            random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
            random_tokens = torch.randint(0, vocab_size, (len(random_token_idx[0]),), device=device)
            final_mask = mask_arr & ~random_token_mask & ~unchanged

            #print(mask_arr.squeeze(0).nonzero().flatten().tolist())
            # create selection from mask_arr
            selection = final_mask.squeeze(0).nonzero().flatten().to(device)
            #print(selection)
            tokens_to_predict = mask_arr.squeeze(0).nonzero().flatten().to(device)
            #print(selection)
            #print(tokenized_sentence.shape)
            # apply selection index to inputs.input_ids, adding MASK tokens
            tokenized_masked_sentence = tokenized_sentence.clone().to(device)
            tokenized_masked_sentence[selection] = tokenizer.mask_token_id
            tokenized_masked_sentence[random_token_idx] = random_tokens

            #print(tokenized_masked_sentence)
            #print(tokenized_sentence)
            #print('\n\n')
            #print('input of the model',tokenized_masked_sentence.shape)
            tokenized_masked_sentence = tokenized_masked_sentence.unsqueeze(0)



            #TODO:
            # Make sure that the input lenght is random from 0.1 to 0.8 of the original sentence
            # This way the model learns to predict the context of a short sentence
            # To test this, load the dataset from ds_test instead of ds3





            out = longformerMLM(tokenized_masked_sentence).logits.squeeze(0)
            #print('-> out:', out.shape)

            prediction = torch.argmax(out, dim=1)
            tokens_predicted = tokenizer.convert_ids_to_tokens(prediction)
            #print('-> prediction:', tokenizer.convert_tokens_to_string(tokens_predicted))
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
            #print('True labels',tokenizer.convert_ids_to_tokens(labels))
            masked_num = labels.shape[0]
            
            #labels[tokenized_sentence != tokenizer.mask_token_id] = -100 # only calculate loss on masked tokens
            #print(masked_toks.shape)
            #print(masked_toks)
            #one_hot_encoding = torch.zeros((masked_num, vocab_size)).scatter_(1, labels.unsqueeze(1), 1).to(device)
            #print(masked_toks.shape)
            #print(one_hot_encoding)

            only_masked_predictions = out[tokens_to_predict]
            #print('Pred labels',tokenizer.convert_ids_to_tokens(torch.argmax(only_masked_predictions, dim = 1)))
            loss = F.cross_entropy(only_masked_predictions, labels)
            #print(loss)
            #print(masked_toks.shape)
            #print(one_hot_encoding.shape)
            if(use_wandb):
                wandb.log({'loss': loss})
            #print(loss)

            
            loss = loss / virtualBatch
            loss.backward() 
            
            
            #torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
            
            if(epoch % virtualBatch == 0):
                optim.step() 
                optim.zero_grad()
            
            #print(tokenizer.convert_ids_to_tokens(prediction))
            if(epoch % partial_save_every == 0):
                torch.save(longformerMLM.state_dict(), save_path + f'longformer_partial_epoch.pth')
            epoch += 1

            #if(epoch > stop_after):
            #    break


        
        torch.save(longformerMLM.state_dict(), save_path + f'longformer_epoch{i}.pth')


if __name__ == "__main__":
    main()