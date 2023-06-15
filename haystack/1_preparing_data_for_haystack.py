import os
from tqdm import tqdm
import pickle 

'''
AS IN HAYSTACK DOCUMENTATION:

docs = [
    {
        'content': DOCUMENT_TEXT_HERE,
        'meta': {'name': DOCUMENT_NAME, ...}
    }, ...
]

'''


'''
MY INITIAL DATASET STRUCTURE:
{'text': text, 'id': doc_name, 'location':location, 'link': link, 'counter': global_id}

'''

# Dataset where the raw documents are stored (text extacted from pdfs)
raw_documents_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/crawler/datasets/raw_pdfs/"
# Dataset where the documents will be saved in the format required by haystack
dataset_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/haystack/raw_for_haystack/"
#dataset_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/test/"
# Read the data
def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        dataset.append(file_path)
    return dataset

raw_ds = loadData(raw_documents_folder,'')

for document in tqdm(raw_ds):
    #print(document)  
    with open(document, 'rb') as f:
        data = pickle.load(f)
        name_id = data['id']
        content = data['text']
        # REPLACE \n WITH WHITESPACE
        content = content.replace('\t', '   ').replace('\n', '   ').replace('\r', '   ')

        meta = {'code': data['id'], 'location': data['location'], 'link': data['link'], 'counter': data['counter']}
        with open(dataset_folder + f'document_{name_id}', 'wb') as f:
          pickle.dump({'content': content, 'meta': meta}, f)
               
                

    