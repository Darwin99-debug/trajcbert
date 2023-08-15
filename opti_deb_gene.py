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

#directories :
#-------------

#loading
data_dir = '/home/daril_kw/data/02.06.23/train_clean.json'
#saving
tokenizer_dir = '/home/daril_kw/data/tokenizer_final'
data_format_dir = '/home/daril_kw/data/data_with_time_info_ok_opti.json'
model_dir = '/home/daril_kw/data/model_resized_embeddings_test'

def truncation_rows(df, nb_rows):
    return df[:nb_rows]

def add_tokenization_column(df, config):
    """Add a column with the tokenization of the POLYLINE column"""
    df['Tokenization_2'] = df['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][0], x[i][1], config) for i in range(len(x))])
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


def main():

    h3_config_size = 10

    with open(data_dir, 'r') as openfile:
        json_loaded = json.load(openfile)

    data_format = pd.DataFrame(data=json_loaded)
    data_format = truncation_rows(data_format, 15)
    data_format = add_tokenization_column(data_format, h3_config_size)
    data_format = extract_time_info(data_format)
    data_format = data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP'], axis=1)
    data_format = formatting_to_str(data_format, 'TAXI_ID')
    data_format = call_type_to_nb(data_format)
    data_format = formatting_to_str(data_format, 'CALL_TYPE')

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    tokenizer, nb_token_geo = add_geo_and_context_tokens_tokenizer(tokenizer, data_format)
    nb_labels = nb_token_geo + 1

    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=nb_labels)
    model.resize_token_embeddings(len(tokenizer))

    #saving the data, the tokenizer and the model
    data_format.to_json(data_format_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir)

if __name__ == "__main__":
    main()