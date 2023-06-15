import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle



current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/longformer/'

with open(current_folder + 'data/newEmbedsFullEpoch.pickle', 'rb') as file:
    data = pickle.load(file)

embeddings = np.array(data['embeddings'])
labels = range(len(embeddings))

print(embeddings[0])
tsne = TSNE(n_components=2, random_state=42)

embedding_tsne = tsne.fit_transform(embeddings)

plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], c=labels)
plt.scatter(embedding_tsne[124, 0], embedding_tsne[124, 1], c='r', label='127')

if(False):
    for i in range(len(embedding_tsne)):
        plt.annotate(str(i+3), (embedding_tsne[i, 0], embedding_tsne[i, 1]))



plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
