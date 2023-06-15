from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/longformer/'

with open(current_folder + 'data/LF88886.pickle', 'rb') as file:
    data = pickle.load(file)

embeddings = np.array(data['embeddings'])
#print(embeddings[10])
title = np.array(data['raw'])
ids = np.array(data['id'])
labels = range(len(embeddings))


n_clusters = 40
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)

predictions = kmeans.predict(embeddings)



show_at_least = 3
clusters = []

for pred in range(n_clusters):
    shown = 0
    cluster = []
    for i in range(len(embeddings)):
        #Only from cluster 0
        if predictions[i] == pred:
            print('title: ', title[i].replace('<pad>', '').replace('<s>', '').replace('</s>', ''))
            print('cluster: ', predictions[i])
            print('id: ', ids[i])
            print('\n\n')
            cluster.append(ids[i])
            shown += 1
        if(shown >= show_at_least):
            break
    clusters.append(cluster)


tsne = TSNE(n_components=2, random_state=42)

embedding_tsne = tsne.fit_transform(embeddings)



plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c='b')

cluster_number = 0
for clust in clusters:
    for id in clust:
        plt.annotate((f'   c:{cluster_number} id:{id}'), (embedding_tsne[id-1, 0], embedding_tsne[id-1, 1]), c='r')
    cluster_number += 1

if(True):
    for i in range(len(embedding_tsne)):
        plt.annotate(f'{str(ids[i])}', (embedding_tsne[i, 0], embedding_tsne[i, 1]))

plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

plt.savefig(current_folder + 'data/clusters88886.png') 
plt.show()