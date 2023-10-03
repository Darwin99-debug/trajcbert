import json
import datetime
import os
import pickle
import h3
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel



def truncation_rows(df, nb_rows):
    return df[:nb_rows]

def add_tokenization_column(df, config):
    """Add a column with the tokenization of the POLYLINE column
    /!\ in that case, for the json file, the trajectories are in the form given by the kaggle dataset /!\
    /!\ ie longitude, latitude instead of latitude, longitude.                                        /!\
    /!\ This is why we have to reverse the order of the coordinates for the tokenization              /!\ """

    df['Tokenization_2'] = df['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][1], x[i][0], config) for i in range(len(x))])
    return df

def extract_time_info(df):
    """Add columns with the day, hour and week of the year knowing the timestamp"""
    df['DATE'] = df['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    df['DAY'] = df['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[2]))
    df['HOUR'] = df['DATE'].apply(lambda x: x.split(' ')[1].split(':')[0])
    df['WEEK'] = df['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1]))
    return df



def formatting_to_str(df, column):
    """Transform the column to string type"""
    if isinstance(df[column][0], str):
        return df
    df[column] = df[column].astype(str)
    return df

def call_type_to_nb(df):
    """Transform the column CALL_TYPE to a number"""
    df['CALL_TYPE'] = df['CALL_TYPE'].apply(lambda x: 0 if x == 'A' else (1 if x == 'B' else 2))
    return df


def add_geo_and_context_tokens_tokenizer(tokenizer, data_format):
    # Add geo tokens to the tokenizer
    liste_token_geo = {token for sublist in data_format['Tokenization_2'] for token in sublist}
    nb_token_geo = len(liste_token_geo)
    tokenizer.add_tokens(list(liste_token_geo))  # Convert set to list and add to tokenizer

    # Add contextual info tokens to the tokenizer
    contextual_info_token = {str(data_format['CALL_TYPE'][i])
                             for i in range(len(data_format))}
    contextual_info_token.update(str(data_format['TAXI_ID'][i])
                                 for i in range(len(data_format)))
    contextual_info_token.update(data_format['DAY'][i]
                                 for i in range(len(data_format)))
    contextual_info_token.update(data_format['HOUR'][i]
                                 for i in range(len(data_format)))
    contextual_info_token.update(data_format['WEEK'][i]
                                 for i in range(len(data_format)))

    tokenizer.add_tokens(list(contextual_info_token))  # Convert set to list and add to tokenizer

    return tokenizer, nb_token_geo



def add_spaces_for_concat(data_format, column):
    """Add spaces before and after the values of the column"""

    #We add space before and after the values of the column because we want to separate the tokens (words) with spaces like that : [CLS] 0 1 2 3 4 5 6 7 8 9 10 [SEP]
    data_format[column]=data_format[column].apply(lambda x: ' '+x)
    return data_format

def get_deb_traj(data_format, len_context_info):
    """the DEB_TRAJ column will be the tokenization column without the last token and the target token"""
    data_format['DEB_TRAJ']=data_format['Tokenization_2'].apply(lambda x: x[:-2])

    # we manage the length of the CONTEXT_INPUT column so that after the concatenation, it does not exceed 512 tokens
    # the -2 corresponds to the two special tokens [CLS] and [SEP]
    # for exemple here, if the trajectory input is too long, we keep the 512-6-2=504 last tokens
    data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: x[-(512-len_context_info-2):] if len(x)>512-len_context_info-2 else x)

    #then we keep the column in form of a string with spaces between the tokens (the space replaces the comma)
    data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: ' '.join(x))

    return data_format



def get_deb_traj_and_target(data_format):
    """Get the DEB_TRAJ and TARGET columns well formatted but without the special tokens [CLS] and [SEP]"""

    #adding spaces for the concatenation after : we want to sperarate the tokens (words) with spaces
    data_format = add_spaces_for_concat(data_format, 'HOUR')
    data_format = add_spaces_for_concat(data_format, 'WEEK')
    data_format = add_spaces_for_concat(data_format, 'CALL_TYPE')
    data_format = add_spaces_for_concat(data_format, 'TAXI_ID')
    data_format = add_spaces_for_concat(data_format, 'DAY')

    #the column CONTEXT_INPUT will be the concatenation of the last token of the tokenization column + the day + the hour + the week + the call type + the taxi id
    data_format['CONTEXT_INPUT'] =data_format['Tokenization_2'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']
    #we get the length of the containing information of the CONTEXT_INPUT column
    len_context_info = len(data_format['CONTEXT_INPUT'][0].split(' '))

    #we get the DEB_TRAJ column
    data_format=get_deb_traj(data_format, len_context_info)

    #we get the TARGET column
    data_format['TARGET']=data_format['Tokenization_2'].apply(lambda x: x[-2])

    return data_format








    #we load the data
with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
    json_loaded = json.load(openfile)
data_format = pd.DataFrame(data=json_loaded)

nb_rows= 20

data_format = truncation_rows(data_format, nb_rows)

    #we count the number of rows for which teh column NB_POINTS is equal to 0 : there are 0 rows
    #>>>print("nombre de lignes pour lesquelles le nombre de points est inférieur à 3 : ", len(data_format[data_format['Nb_points']<3]))
    #   nombre de lignes pour lesquelles le nombre de points est inférieur à 3 :  0


    #we add the tokenization column
data_format = add_tokenization_column(data_format, 10)

    #we add the time info columns
data_format = extract_time_info(data_format)

    #we remove the useless columns
data_format = data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP'], axis=1)

    #we transform the columns TAXI_ID which was a number to string type
data_format = formatting_to_str(data_format, 'TAXI_ID')

    #we transform the column CALL_TYPE to a number instead of a letter
data_format = call_type_to_nb(data_format)

    #we transform the column CALL_TYPE which was a number to string type
data_format = formatting_to_str(data_format, 'CALL_TYPE')


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    #we add the geographical and contextual tokens to the tokenizer so that the vocabulary of the tokenizer is adapted to our data    
tokenizer, nb_token_geo = add_geo_and_context_tokens_tokenizer(tokenizer, data_format)
    #we get the number of labels which is the number of geographical tokens + 1 (the +1 is for the [SEP] token which is for the end of the sequence and the prediction)
nb_labels = nb_token_geo + 1

    #we get the model from the HuggingFace library, this one is the model bert-base-cased but we could have taken the non trained model (if we want to train it from scratch)
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=nb_labels)
    #we add the geographical and contextual tokens to the model so that the size of the model`s embedding is adapted to our data
model.resize_token_embeddings(len(tokenizer))


print("nb classes:")
print(nb_labels)

data_format = get_deb_traj_and_target(data_format)

