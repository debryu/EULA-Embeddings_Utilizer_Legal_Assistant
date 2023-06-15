from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
import os
from tqdm import tqdm
import pickle
from haystack.nodes import PreProcessor
from rank_bm25 import BM25Okapi

dataset_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/haystack/raw_for_haystack/"
#dataset_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/haystack/dataset/test/"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/haystack/raw_for_bm25/"

# Again initialize the preprocess and the converter for the documents (just to parse words)
converter = TextConverter(remove_numeric_tables=True, valid_languages=["it"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=1,
    split_respect_sentence_boundary=False,
)

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
tokenized_documents = []
documents = []
ids = []
for i,document in tqdm(enumerate(raw_ds)): 
    with open(document, 'rb') as f:
        data = pickle.load(f)

    # Retrieve the id and the content of the document
    name_id = data['meta']['code']
    content = data['content']
    
    documents.append(content + f'-ID%:{name_id}')
    doc_txt = converter.convert(file_path=document, meta=None)[0]
    docs_default = preprocessor.process([doc_txt])
    words = [word.content for word in docs_default]
    tokenized_documents.append(words)
    ids.append(name_id)

    print(i)
    
    # Stop at 10000 documents because it is taking so long
    if(i==10000):
        break

# Save the documents as they are but also the tokenized version (where tokenized means that the document is split by words)
with open(save_path + f'10000_INDEX.pickle', 'wb') as f:
          pickle.dump({'docs': documents, 'tok_docs': tokenized_documents}, f)

# Just to check if everything is ok
bm25 = BM25Okapi(tokenized_documents)
# Try to query 
query = 'Non mi è stato riconosciuto il titolo di laurea magistrale conseguito in Germania. Vorrei presentare ricorso.'
tokenized_query = query.split(" ") 

# Check the results
results = bm25.get_top_n(tokenized_query, documents, n=2)

print(results[0])

