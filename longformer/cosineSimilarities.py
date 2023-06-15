import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import pandas as pd



current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/longformer/'

with open(current_folder + 'data/embedsFullEpoch.pickle', 'rb') as file:
    data = pickle.load(file)
  
embeddings = np.array(data['embeddings'])
title = np.array(data['raw'])
ids = np.array(data['id'])
labels = range(len(embeddings))

# Create a pandas dataframe containing embeddings and titles
df = pd.DataFrame(embeddings)
df['title'] = title
df['id'] = ids

# Show the dataframe 
print(df.head(10))