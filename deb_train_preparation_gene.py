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
def truncation_rows(df, nb_rows):
    return df[:nb_rows]
data_format = truncation_rows(data_format, 60)

#we create the correct tokenization column
def add_tokenization_column(df):
    df['Tokenization_2'] = df['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][0], x[i][1], 10) for i in range(len(x))])
    return df

data_format = add_tokenization_column(data_format)

#on récupère la date à partir du timestamp
def extract_time_info(df):
    df['DATE'] = df['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    df['DAY'] = df['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[2]))
    df['HOUR'] = df['DATE'].apply(lambda x: x.split(' ')[1].split(':')[0])
    df['WEEK'] = df['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1]))
    return df

data_format = extract_time_info(data_format)

#we remove the useless columns
data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP' ],axis=1,inplace=True)



#we put the data in a str format
data_format['CALL_TYPE'] = data_format['CALL_TYPE'].apply(lambda x: str(1) if x=='A' else str(2) if x=='B' else str(3))
def formatting_to_str(df, column):
    if type(df[column][0])!=str:
        df[column]=df[column].apply(lambda x: str(x))
    return df

print("we put in the right format")
data_format = formatting_to_str(data_format, 'TAXI_ID')
print("gestion du tokenizer commencée")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def add_geo_and_context_tokens_tokenizer(tokenizer):
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
    return tokenizer, nb_token_geo



print("On a le tokenizer final")

#On a besoin du nombre de labels, celui-ci correspond au nombre de tokens géographiques + 1 (pour le token [SEQ] indiquant la fin de la séquence)
tokenizer, nb_token_geo = add_geo_and_context_tokens_tokenizer(tokenizer)
nb_labels = nb_token_geo + 1


model=BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=nb_labels)
#on adapte la taille de l'embedding pour qu'elle corresponde au nombre de tokens géographiques + 1
model.resize_token_embeddings(len(tokenizer))



#save the data_format here : '/home/daril_kw/data/data_with_time_info_ok.json'
data_format.to_json('/home/daril_kw/data/data_with_time_info_ok.json')

#save the tokenizer here : '/home/daril_kw/data/tokenizer_final'
tokenizer.save_pretrained('/home/daril_kw/data/tokenizer_final')
