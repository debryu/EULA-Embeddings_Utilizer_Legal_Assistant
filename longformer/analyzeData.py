import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle


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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

checkpoint = 'longformerNewDifferentDataset_partial_epoch88886.pth'
model_name = "allenai/longformer-base-4096"
save_path = "C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/ANLP_project/ANLP/saved_models/MLM/longformer-base-4096/epochs/"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = len(tokenizer)
print('Vocab tokenizer size', vocab_size)
#tokenized_content = tokenizer.tokenize("my name is earl", return_tensors='pt')
#print(tokenized_content)

from transformers import AutoModel, AutoModelForMaskedLM
longformerMLM = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
longformerMLM.to(device)


longformerMLM.load_state_dict(torch.load(save_path + checkpoint, map_location=torch.device(device)))
print('Model loaded!')

longformerMLM.eval()

def getEmbedding(text):
    tokenized_text = torch.tensor(tokenizer.encode(text, return_tensors='pt').to(device))
    output = longformerMLM(tokenized_text)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token

def getEmbeddingFromTokens(tokenized_text):
    output = longformerMLM(tokenized_text)
    #print(output)
    last_layer = output['hidden_states'][-1].squeeze(0)
    cls_token = last_layer[0].squeeze(0)
    return cls_token


#query_embs = getEmbedding('annullamento della graduatoria per posto da dipendente pubblico. dolendosi di non essere stato ammesso alla prova orale per essere stata la sua prova scritta illegittimamente valutata con un punteggio pu essere esaminato immediatamente il merito del ricorso, siccome infondato;e) con il primo motivo, parte ricorrente lamenta che erroneamente sarebbero state considerate errate le risposte a tre quesiti della prova scritta, somministrata attraverso dei quiz a risposta multipla; tuttavia:- e1) in materia di pubblico impiego, ai sensi del d.lgs. 30 marzo 2001, n. 165, la contestazione scritta delle infrazioni  propedeutica allirrogazione di sanzioni superiori al rimprovero verbale; dunque,  errata la risposta no data dal ricorrente alla domanda n. 15, a norma del D.lgs 165/2001 il capo della struttura in cui il dipendente pubblico lavora pu adottare nei confronti di questultimo il provvedimento del rimprovero verbale, senza previa tempestiva contestazione scritta?;- e2) a mente dellart. 5, comma 1, lett. b) d.lgs. 3 aprile 2006, n. 152, la VIA (valutazione di impatto ambientale)').to('cpu').tolist()
query_embs = getEmbedding('annullamento della graduatoria relativa al bando di concorso pubblico per titoli ed esami').to('cpu').detach().numpy()

ttext = 'Con l�appello in esame gli odierni appellanti, nella qualit� rispettivamente di proprietario e di conduttore dell�immobile interessato, impugnavano la sentenza n. 4437 del 2015 del Tar Campania, recante rigetto dell�originario gravame. Quest�ultimo era stato proposto dalle stesse parti  al fine di ottenere l�annullamento dell�ordinanza n. 245 del 23 maggio 2005, con la quale l�amministrazione comunale ha ingiunto la demolizione delle opere edificate sul suddetto terreno, sostanziatesi nella realizzazione di una strada di accesso in conglomerato bituminoso con installazione di un cancello in ferro all�ingresso, nel livellamento dell�area a monte adibita a deposito di materiali edili, nella installazione di una baracca di cantiere con base in cemento adibita a deposito e di un container adibito ad officina, nonch� nella edificazione di una tettoia in putrelle in ferro (�imbullonate al suolo�) e lamiera adibita a ricovero di autoveicoli. Nel ricostruire in fatto e nei documenti la vicenda, parte appellante formulava, contestando le argomentazioni svolte nella sentenza impugnata, i seguenti motivi di appello: - violazione degli artt. 24 Cost., d.P.R. 380 del 2001, d.lgs. 42 del 2004 e l. 64 del 1974, illogicit�, difetto di istruttoria e di motivazione, per genericit� degli elementi indicati dal Comune; - analoghi vizi in ordine alla qualificazione delle opere abusive, trattandosi di opere minori e pertinenziali, sanabili e soggette a sanzione pecuniaria; - analoghi vizi in merito all�assenza di vincolativit� trattandosi di interventi di mera manutenzione o pertinenziali; - omessa valutazione della presentazione della domanda di sanatoria; - mancata indicazione dell�area destinata ad essere acquisita al patrimonio comunale in caso di inottemperanza; - violazione delle garanzie partecipative. La parte appellata si costituiva in giudizio chiedendo il rigetto dell�appello. Alla pubblica udienza del 22 settembre 2022 la causa passava in decisione. DIRITTO 1. L�appello � destituito di fondamento, in termini tali da rendere applicabile l�art. 74 cod proc amm, in quanto i diversi motivi si scontrano con i consolidati orientamenti di questo Consiglio. 2. In termini fattuali � pacifico, in quanto emergente dagli atti e confermato dalla narrativa in fatto dello stesso atto di appello, che i manufatti oggetto di demolizione siano quelli accertati dal Comune come abusivamente realizzati. 3. In relazione al primo motivo di appello, va ribadito il carattere reale delle sanzioni in materia edilizia, nel senso che il presupposto per l adozione di un ordinanza di ripristino � non gi� l accertamento di responsabilit� nella commissione dell illecito, ma l esistenza d una situazione dei luoghi contrastante con quella prevista nella strumentazione urbanistico-edilizia. La sanzione ripristinatoria costituisce atto vincolato, per la cui adozione non � necessaria la valutazione specifica delle ragioni di interesse pubblico, n� la comparazione di questi con gli interessi privati coinvolti, n� tantomeno una motivazione sulla sussistenza di un interesse pubblico concreto ed attuale alla demolizione, non essendo in alcun modo ammissibile l esistenza di un affidamento tutelabile alla conservazione di una situazione di fatto abusiva (cfr. ex multis Consiglio di Stato, sez. VI, 17 luglio 2018, n. 4368 e 3 gennaio 2022, n. 10).  In definitiva, l�ordine di demolizione, avendo natura di atto vincolato, non necessita di relativa motivazione anche se contiene una motivazione adeguata nel momento in cui � come nel caso di specie - descrive gli interventi abusivamente effettuati. 4. In relazione al secondo ed al terzo motivo di appello, da esaminare congiuntamente in quanto entrambi tesi a contestare la qualificazione degli abusi, appaiono corrette le conclusioni fatte proprie dall�amministrazione e condivise dal Giudice di prime cure. 4.1 Le opere abusive accertate, realizzate in zona vincolata, hanno dato luogo ad un intervento di rilevante impatto, correttamente considerato in termini unitari anche a fronte della incisivit� su di un�area soggetta a specifica tutela, come desumibile dalla chiara ricostruzione posta a base della statuizione contestata: realizzazione di una strada di accesso in conglomerato bituminoso con installazione di un cancello in ferro all�ingresso, nel livellamento dell�area a monte adibita a deposito di materiali edili; installazione di una baracca di cantiere con base in cemento adibita a deposito e di un container adibito ad officina; edificazione di una tettoia in putrelle in ferro (�imbullonate al suolo�) e lamiera adibita a ricovero di autoveicoli. 4.2 Sulla scorta di tali risultanze, va ribadito che l�opera edilizia abusivamente eseguita va identificata con riferimento all unitariet� dell immobile o del complesso immobiliare ove realizzato in esecuzione di un disegno unitario, come nella specie. In linea generale, infatti, al fine di valutare l incidenza sull assetto del territorio di un intervento edilizio, consistente in una pluralit� di opere, va compiuto � specie in ambito soggetto a specifica tutela vincolistica - un apprezzamento globale, atteso che la considerazione atomistica dei singoli interventi non consente di comprenderne in modo adeguato limpatto effettivo complessivo, con la conseguenza che i molteplici interventi eseguiti non vanno considerati, dunque, in maniera �frazionata� (cfr. ad es. Consiglio di Stato, sez. VI, 08/09/2021, n. 6235). 4.3 Nel caso di specie la valutazione svolta dal Comune e condensata nella motivazione dell'
ttext = "diniego del visto d'ingresso per motivi di turismo"
ttext = "diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo diniego del visto d'ingresso per motivi di turismo   "
#ttext = "ha approvato il piano di lottizzazione (denominato Angelini) nella localit� marina di Mesu Turrj, con riserva di provvedere con successivi deliberati alla vendita dei singoli lotti e alla determinazione del prezzo e delle modalit� di costruzione dei fabbricati."
#ttext = "Il ricorrente lamenta l�esclusione dagli elenchi aggiuntivi relativi alla prima fascia delle graduatorie provinciali per le supplenze (GPS), disposta sulla scorta del fatto che non ha conseguito, alla data del 31 luglio 2021, il riconoscimento della qualifica professionale di docente conseguita all�estero, come prescritto dal d.m. n. 51 del 2021."
query_embs = getEmbedding(ttext).to('cpu').detach().numpy()

uni = getEmbedding('università').to('cpu').detach().numpy()
lavoro = getEmbedding('lavoro').to('cpu').detach().numpy()
estero = getEmbedding('estero').to('cpu').detach().numpy()
famiglia = getEmbedding('famiglia').to('cpu').detach().numpy()
immigrazione = getEmbedding('immigrazione').to('cpu').detach().numpy()
sanita = getEmbedding('sanità').to('cpu').detach().numpy()
tributario = getEmbedding('tributario').to('cpu').detach().numpy()
edilizia = getEmbedding('edilizia').to('cpu').detach().numpy()




current_folder = 'C:/Users/debryu/Desktop/VS_CODE/HOME/ANLP/all_datasets/data/'

with open(current_folder + 'LF88886.pickle', 'rb') as file:
    data = pickle.load(file)

embeddings = np.array(data['embeddings'])
#print(embeddings[10])
title = np.array(data['raw'])
ids = np.array(data['id'])
labels = range(len(embeddings)+1+7)

file = open(current_folder + '/data/legend88886.txt' , "w")
for i in range(len(embeddings)):
    try:
        file.write(f'{ids[i]}: {title[i]}\n\n')
    except:
        file.write(f'{ids[i]}: ERROR\n\n')
    

file.close()



#Add the query embedding to the embeddings
embeddings = np.vstack((embeddings, query_embs))
embeddings = np.vstack((embeddings, uni))
embeddings = np.vstack((embeddings, lavoro))
embeddings = np.vstack((embeddings, estero))
embeddings = np.vstack((embeddings, famiglia))
embeddings = np.vstack((embeddings, immigrazione))
embeddings = np.vstack((embeddings, sanita))
embeddings = np.vstack((embeddings, tributario))
embeddings = np.vstack((embeddings, edilizia))

#print(embeddings)
#Add the query title to the titles
title = np.append(title, 'query')
title = np.append(title, 'università')
title = np.append(title, 'lavoro')
title = np.append(title, 'estero')
title = np.append(title, 'famiglia')
title = np.append(title, 'immigrazione')
title = np.append(title, 'sanità')
title = np.append(title, 'tributario')
title = np.append(title, 'edilizia')


#Add the query id to the ids
ids = np.append(ids, 'query')
ids = np.append(ids, 'università')
ids = np.append(ids, 'lavoro')
ids = np.append(ids, 'estero')
ids = np.append(ids, 'famiglia')
ids = np.append(ids, 'immigrazione')
ids = np.append(ids, 'sanità')
ids = np.append(ids, 'tributario')
ids = np.append(ids, 'edilizia')

#print(ids)
#print(title)


# Define the cosine similarity function
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


topics =  [uni, lavoro, estero, famiglia, immigrazione, sanita, tributario, edilizia]
# Compute the cosine similarity between the query and all the embeddings
cosine_similarities = [cosine_similarity(query_embs,topics[i]) for i in range(len(topics))]
print(cosine_similarities)

tsne = TSNE(n_components=2, random_state=42)

embedding_tsne = tsne.fit_transform(embeddings)

plt.scatter(embedding_tsne[:-8, 0], embedding_tsne[:-8, 1], c='b')
plt.scatter(embedding_tsne[9, 0], embedding_tsne[9, 1], c='r', label='10')
plt.scatter(embedding_tsne[-9, 0], embedding_tsne[-9, 1], c='g', label='query')
plt.scatter(embedding_tsne[-8, 0], embedding_tsne[-8, 1], c='y', label='università')
plt.scatter(embedding_tsne[-7, 0], embedding_tsne[-7, 1], c='m', label='lavoro')
plt.scatter(embedding_tsne[-6, 0], embedding_tsne[-6, 1], c='r', label='estero')
plt.scatter(embedding_tsne[-5, 0], embedding_tsne[-5, 1], c='g', label='famiglia')
plt.scatter(embedding_tsne[-4, 0], embedding_tsne[-4, 1], c='grey', label='immigrazione')
plt.scatter(embedding_tsne[-3, 0], embedding_tsne[-3, 1], c='r', label='sanità')
plt.scatter(embedding_tsne[-2, 0], embedding_tsne[-2, 1], c='r', label='tributario')
plt.scatter(embedding_tsne[-1, 0], embedding_tsne[-1, 1], c='r', label='edilizia')



if(True):
    for i in range(len(embedding_tsne)):
        plt.annotate(str(ids[i]), (embedding_tsne[i, 0], embedding_tsne[i, 1]))



plt.title("t-SNE Visualization of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()

plt.savefig(current_folder + 'data/graph88886.png') 
plt.show()