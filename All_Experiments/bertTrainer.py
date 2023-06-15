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


loadModel = False
dataset_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/dataset/MLM/ds1/'
batch_size = 1
#bert_lr = 0.00000000000001
#mlm_head_lr = 0.000001
bert_lr = 0.00001
mlm_head_lr = 0.1

use_wandb = False

iterations = 100
partial_save_every = 1000
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

model = BertModel.from_pretrained(model_name).to(device)
'''
print(device)
unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased', device=device)
print(unmasker("Hello I'm a [MASK] model."))


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
text = "Hello I'm a [MASK] model."

encoded_input = tokenizer(text, return_tensors='pt').to(device)
print(encoded_input)

print(tokenizer.convert_ids_to_tokens(103))
print(tokenizer.convert_tokens_to_ids('[MASK]'))
print(tokenizer.convert_ids_to_tokens(102))
print(tokenizer.convert_tokens_to_ids('[SEP]'))
output = model(**encoded_input)

#print(output.shape)
'''

wandbConfig = {
 
  "version": "1.0",
  "model used": model_name,
  "batch size": batch_size,
  "learning rate mlm": mlm_head_lr,
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
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), len(ds)

    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,  drop_last=True), len(ds)



train_ds = loadData(dataset_folder, 'train/')
dataset_train, dataset_train_length = create_data_loader(train_ds, 1, eval=True)



class MLM_head(nn.Module):
    def __init__(self):
        super(MLM_head, self).__init__()    
        self.FC = nn.Sequential(  
                                    # 768
                                    nn.LayerNorm(768), 
                                    nn.Linear(768, 768*4),
                                    nn.GELU(),
                                    #nn.Dropout(0.2),
                                    nn.Linear(768*4, vocab_size),
                                    )
        
        self.FC2 = nn.Sequential(  
                                    # 768
                                    nn.LayerNorm(768), 
                                    nn.Linear(768, 768),
                                    nn.GELU(),
                                    nn.Linear(768, 768),
                                    nn.Dropout(0.2),
                                    nn.GELU(),
                                    nn.Linear(768, 768),
                                    nn.GELU(),
                                    nn.Linear(768, 768*4),
                                    nn.GELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(768*4, vocab_size),
                                    )
        self.FC3 = nn.Sequential(
                                    nn.LayerNorm(768), 
                                    nn.Linear(768, vocab_size),
                                    )
        self.FC4 = nn.Sequential(
                                    nn.LayerNorm(768), 
                                    nn.Linear(768,512),
                                    nn.GELU(),
                                    nn.Linear(512,512),
                                    nn.GELU(),
                                    nn.Linear(512, vocab_size),
                                    nn.Tanh(),
                                    #nn.Softmax(),
                                    )
        self.FC5 = nn.Sequential(
                                    nn.LayerNorm(768), 
                                    nn.Linear(768, vocab_size),
                                    nn.Tanh(),
                                    #nn.Softmax(),
                                    )
    def forward (self, x):
        #print(x.shape)
        x = self.FC4(x)
        #print(x.shape)
        pred = torch.argmax(x, dim = 1)
        #print('pred: ', pred)
        return x,pred
  
from transformers import BertForMaskedLM
bertMLM = BertForMaskedLM.from_pretrained(model_name)
bertMLM.to(device)

MLM_classifier = MLM_head()
MLM_classifier.to(device)
MLM_classifier.train()
model.to(device)
model.train()

if(loadModel):
    model.load_state_dict(torch.load(save_path + 'bert_partial_epoch0.pth', map_location=torch.device(device)))
    MLM_classifier.load_state_dict(torch.load(save_path + 'MLM_head_partial_epoch0.pth', map_location=torch.device(device)))
    print('Model loaded!')

MLM_head_parameters = MLM_classifier.parameters()
bert_parameters = model.parameters()

optim = torch.optim.AdamW([
    {'params': bert_parameters, 'lr': bert_lr},
    {'params': MLM_head_parameters, 'lr': mlm_head_lr},
    ])
    
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
        mask_arr = rand < 0.15 * (tokenized_sentence != 101) * (tokenized_sentence != 102) * (tokenized_sentence != 117) * (tokenized_sentence != 119)
        unchanged = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.6)
        random_token_mask = mask_arr & (torch.rand(mask_arr.shape, device=device) < 0.3)
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
        bert_output = model(tokenized_sentence)['last_hidden_state'].squeeze(0)
        bo = bertMLM(tokenized_sentence).logits.squeeze(0)
        probs = F.softmax(bo, dim=1)
        print('bert-out:', bo)
        print(bo.shape)
        #print('bert output', bert_output)
        #print(bert_output.shape)
        prediction,pred_tok = MLM_classifier(bert_output)
        #print('predictions', tokenizer.convert_ids_to_tokens(pred_tok))
        #print('preds:',prediction.shape)
        #print(prediction.shape)
        #logits = F.softmax(prediction, dim=1)   #ALREADY DOING THAT IN THE LAYER
        masked_toks = prediction[tokens_to_predict,:].to(device)
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
        print('Pred labels',tokenizer.convert_ids_to_tokens(torch.argmax(probs, dim = 1)))
        loss = F.cross_entropy(masked_toks, labels)
        #print(loss)
        #print(masked_toks.shape)
        #print(one_hot_encoding.shape)
        if(use_wandb):
           wandb.log({'loss': loss})
        #print(loss)
        optim.zero_grad()
        loss.backward() 
        #torch.nn.utils.clip_grad_norm_(bxfinder.parameters(), clipping_value)
        optim.step() 
        
        #print(tokenizer.convert_ids_to_tokens(prediction))
        if(epoch % partial_save_every == 0):
            torch.save(model.state_dict(), save_path + f'bert_partial_epoch.pth')
            torch.save(MLM_classifier.state_dict(), save_path + f'MLM_head_partial_epoch.pth')
        epoch += 1

        break
    break
    torch.save(model.state_dict(), save_path + f'bert_epoch{i}.pth')
    torch.save(MLM_classifier.state_dict(), save_path + f'MLM_head_epoch{i}.pth')