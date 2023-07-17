from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
import random
import json
import time
import os
import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
import h3

with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:

    # Reading from json file
    json_loaded = json.load(openfile)

print("We put the data in a dataset.")
 
#we put them in a dataframe
data_format = pd.DataFrame(data=json_loaded)

#we keep only 60 rows
data_format = data_format[:60]

#we create the correct tokenization column
data_format['Tokenization_2'] = data_format['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][0], x[i][1], 10) for i in range(len(x))])

#on récupère la date à partir du timestamp
data_format['DATE'] = data_format['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

#a partir de cela on récupère le jour de la semaine sous forme d'un nombre entre 1 et 7 pour lundi à dimanche
data_format['DAY'] = data_format['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[2]))

#ensuite, on récupère l'heure sous forme d'un nombre entre 0 et 23
data_format['HOUR'] = data_format['DATE'].apply(lambda x: x.split(' ')[1].split(':')[0])

#enfin on recupère le numéro de la semaine dans l'année
data_format['WEEK'] = data_format['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1]))



#we remove the useless columns
data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP' ],axis=1,inplace=True)



#we put the data in a str format
print("we put in the right format")
data_format['CALL_TYPE'] = data_format['CALL_TYPE'].apply(lambda x: str(1) if x=='A' else str(2) if x=='B' else str(3))
if type(data_format['TAXI_ID'][0])!=str:
    data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))
#idem pour le call_type
if type(data_format['CALL_TYPE'][0])!=str:
    data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: str(x))

print("gestion du tokenizer commencée")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


liste_token_geo = []

for i in range(len(data_format)):
    for j in range(len(data_format['Tokenization_2'][i])):
        liste_token_geo.append(data_format['Tokenization_2'][i][j])

#on enlève les doublons
liste_token_geo = list(set(liste_token_geo))

#on garde le nombre de tokens géographiques pour la suite
nb_token_geo = len(liste_token_geo)

#On ajoute les tokens géographiques au tokenizer
tokenizer.add_tokens(liste_token_geo)

contextual_info_token = []
for i in range(len(data_format)):
    contextual_info_token.append(data_format['CALL_TYPE'][i])
    contextual_info_token.append(str(data_format['TAXI_ID'][i]))
    contextual_info_token.append(data_format['DAY'][i])
    contextual_info_token.append(data_format['HOUR'][i])
    contextual_info_token.append(data_format['WEEK'][i])
      

#we remove the duplicates
contextual_info_token = list(set(contextual_info_token))
    

#we add the new tokens to the tokenizer 
tokenizer.add_tokens(contextual_info_token)


print("On a le tokenizer final")

#On a besoin du nombre de labels, celui-ci correspond au nombre de tokens géographiques + 1 (pour le token [SEQ] indiquant la fin de la séquence)

nb_labels = nb_token_geo + 1

model=BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=nb_labels)
#on adapte la taille de l'embedding pour qu'elle corresponde au nombre de tokens géographiques + 1
model.resize_token_embeddings(len(tokenizer))



#save the data_format here : '/home/daril_kw/data/data_with_time_info_ok.json'
data_format.to_json('/home/daril_kw/data/data_with_time_info_ok.json')

#save the tokenizer here : '/home/daril_kw/data/tokenizer_final'
tokenizer.save_pretrained('/home/daril_kw/data/tokenizer_final')
