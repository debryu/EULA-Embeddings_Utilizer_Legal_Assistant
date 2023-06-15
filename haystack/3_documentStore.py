
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
import os
from tqdm import tqdm
import pickle

from haystack.nodes import PreProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


''' 

DO EVERYTHING IN ONE STEP

'''
tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-BERTino')
model = SentenceTransformer("efederici/sentence-BERTino").to(device)
converter = TextConverter(remove_numeric_tables=True, valid_languages=["it"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=1,
    split_respect_sentence_boundary=False,
)

raw_documents_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_Crawler/datasets/raw_pdfs/'
#raw_documents_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_Crawler/datasets/test/'
dataset_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/raw_for_haystack/"
#dataset_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/test/"
documentStore_folder = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/index/"
# load all the paths of the documents
def loadData(path,split):
    files =  os.listdir(path + split)
    dataset = []
    for file in files:
        file_path = os.path.join(path+split, file)
        dataset.append(file_path)
    return dataset

# Load the documents
raw_ds = loadData(raw_documents_folder,'')
ds_length = len(raw_ds)


# LOAD ALL THE DOCUMENTS
for i,document in enumerate(raw_ds):
    
    with open(document, 'rb') as f:
        data = pickle.load(f)
    name_id = data['id']
    content = data['text']
    
    # REPLACE \n WITH WHITESPACE
    content = content.replace('\t', '   ').replace('\n', '   ').replace('\r', '   ')
    metadata = {'code': data['id'], 'location': data['location'], 'link': data['link'], 'counter': data['counter']}
    
    haystack_document_path = dataset_folder + f'document_{name_id}'
    doc_txt = converter.convert(haystack_document_path, meta=None)[0]
    parsed_doc = preprocessor.process([doc_txt])
    # Extract only the text
    parsed_doc = [sentence.content for sentence in parsed_doc]
    #print(len(parsed_doc))
    #print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")
    embeddings = model.encode(parsed_doc)
    #print(embeddings.shape)
    
    if i%200 == 0:
        print(f'Processed {i} documents out of {ds_length}')

    with open(documentStore_folder + f'index_{name_id}.pickle', 'wb') as f:
          pickle.dump({'embeddings': embeddings}, f)
    



