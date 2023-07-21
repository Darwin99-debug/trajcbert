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
import torch.distributed as dist
import h3
from sklearn.metrics import f1_score

print("imports done")

WORLD_S=2


with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:

    # Reading from json file
    json_loaded = json.load(openfile)

print("We put the data in a dataset.")
 
#we put them in a dataframe
data_format = pd.DataFrame(data=json_loaded)

#we keep only 60 rows
data_format = data_format[:20]

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
data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL','TRIP_ID', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP' ],axis=1,inplace=True)



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
model.save_pretrained('/home/daril_kw/data/model_before_training')


print("gestion du format de l'input commencée")
#gestion du format de l'input
data_format['HOUR']=data_format['HOUR'].apply(lambda x: ' '+x)
data_format['WEEK']=data_format['WEEK'].apply(lambda x: ' '+x)
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)
data_format['DAY']=data_format['DAY'].apply(lambda x: ' '+x)

data_format['CONTEXT_INPUT'] =data_format['Tokenization_2'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']

len_context_info = len(data_format['CONTEXT_INPUT'][0].split(' '))

#la colonne DEB_TRAJ sera la colonne Tokenization jusqu'a l'avant-dernier token exclu

data_format['DEB_TRAJ']=data_format['Tokenization_2'].apply(lambda x: x[:-2])


# on gère la longueur de la colonne CONTEXT_INPUT pour qu'après la concaténation, elle ne dépasse pas 512 tokens
#le -2 correspond aux deux tokens spéciaux [CLS] et [SEP]
# for exemple here, if the trajectory input is too long, we keep the 512-6-2=504 last tokens
data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: x[-(512-len_context_info-2):] if len(x)>512-len_context_info-2 else x)

#then we keep the column in form of a string
data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: ' '.join(x))

data_format['TARGET']=data_format['Tokenization_2'].apply(lambda x: x[-2])


#on enlève les colonnes inutiles
if 'Tokenization' in data_format.columns:
    data_format.drop(['Tokenization'],axis=1,inplace=True)
if 'CALL_TYPE' in data_format.columns:
    data_format.drop(['CALL_TYPE'],axis=1,inplace=True)
if 'TAXI_ID' in data_format.columns:
    data_format.drop(['TAXI_ID'],axis=1,inplace=True)
if 'DAY' in data_format.columns:
    data_format.drop(['DAY'],axis=1,inplace=True)
if 'HOUR' in data_format.columns:
    data_format.drop(['HOUR'],axis=1,inplace=True)
if 'WEEK' in data_format.columns:
    data_format.drop(['WEEK'],axis=1,inplace=True)
if 'Nb_points_token' in data_format.columns:
    data_format.drop(['Nb_points_token'],axis=1,inplace=True)


c_inputs=data_format.CONTEXT_INPUT.values
traj_inputs=data_format.DEB_TRAJ.values
targets=data_format.TARGET.values

print("concaténation des inputs, padding etc")

input_ids = []
full_inputs = []
attention_masks = []
for i in tqdm(range(len(c_inputs))):
    #no truncation is needed because we managed it before

    #we concatenate the context input and the trajectory input adding manually the CLS token and the SEP token
    full_input = '[CLS] ' + c_inputs[i] + ' ' + traj_inputs[i] + ' [SEP]'
    full_inputs.append(full_input)
    #encoded_c_input=tokenizer.encode(c_inputs[i], add_special_tokens=False)
    #encoded_traj_input=tokenizer.encode(traj_inputs[i], add_special_tokens=False)
    #we add manually the CLS token and the SEP token when we concatenate the two inputs
    #encoded_full_input=[101] + encoded_c_input + encoded_traj_input + [102]
    #the[101] token is the CLS token and the [102] token is the SEP token

    encoded_full_input=tokenizer.encode(full_input, add_special_tokens=False)
    #we pad the input to the maximum length of 512
    encoded_full_input=encoded_full_input + [0]*(512-len(encoded_full_input))
    input_ids.append(encoded_full_input)
    #we create the attention mask
    att_mask = [float(i>0) for i in encoded_full_input]
    attention_masks.append(att_mask)
    #the attention mask is a list of 0 and 1, 0 for the padded tokens and 1 for the other tokens
    #the float(i>0) is 0 if i=0 (ie if the token is a padded token) and 1 if i>0 (ie if the token is not a padded token)




#save the lists full_inputs, inputs_ids, attention_masks and the targets in different files
with open('/home/daril_kw/data/input_ids_20.pkl', 'wb') as fp:
    pickle.dump(input_ids, fp)
with open('/home/daril_kw/data/attention_masks_20.pkl', 'wb') as fp:
    pickle.dump(attention_masks, fp)
with open('/home/daril_kw/data/targets_20.pkl', 'wb') as fp:
    pickle.dump(targets, fp)
with open('/home/daril_kw/data/full_inputs_20.pkl', 'wb') as fp:
    pickle.dump(full_inputs, fp)
    
