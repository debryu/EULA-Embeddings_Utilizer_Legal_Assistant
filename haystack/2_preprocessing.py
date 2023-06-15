from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
import os
from tqdm import tqdm
import pickle
import logging
from haystack.nodes import PreProcessor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# Dataset where the documents prepared for Haystack are stored
dataset_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/haystack/raw_for_haystack/"
# This is not used anymore, since not using haystack for indexing
#save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/haystack_parsing/"
# Load the tokenizer just to check the length of the documents and sentences (in tokens)
tokenizer = AutoTokenizer.from_pretrained('efederici/sentence-BERTino')
# Load the model 
model = SentenceTransformer("efederici/sentence-BERTino")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Use the haystack preprocessing tools to split the documents into sentences
converter = TextConverter(remove_numeric_tables=True, valid_languages=["it"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="sentence",
    split_length=1,
    split_respect_sentence_boundary=False,
)

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# load all the paths of the documents
def loadData(path):
    files =  os.listdir(path)
    dataset = []
    for file in files:
        file_path = os.path.join(path, file)
        dataset.append(file_path)
    return dataset

raw_ds = loadData(dataset_path)

i = 0
lengths = []
for document in tqdm(raw_ds): 
    doc_txt = converter.convert(file_path=document, meta=None)[0]
    docs_default = preprocessor.process([doc_txt])
    #print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")
    for doc in docs_default:
        tokenized = tokenizer.tokenize(doc.content)
        lengths.append(len(tokenized))
        #print(tokenized)
        #print(doc.content)
        #print('\n------------------\n')

    i += 1
    # Just the first 1000 because it takes too long
    if i == 1000:
        break

# Print the max and min length of the sentences
print(f'MAX LEN:{max(lengths)} - MIN LEN:{min(lengths)}')