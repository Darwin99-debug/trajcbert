import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import json



#load the tokenizer from /home/daril_kw/data/tokenizer_final
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_final')
#load the dataset from home/daril_kw/data/data_with_time_info_ok.json
with open('/home/daril_kw/data/data_with_time_info_ok.json', 'r') as openfile:
    json_loaded = json.load(openfile)
    

data_format = pd.DataFrame(data=json_loaded)

def add_spaces_for_concat(data_format, column):
    data_format[column]=data_format[column].apply(lambda x: ' '+x)
    return data_format

data_format = add_spaces_for_concat(data_format, 'HOUR')
data_format = add_spaces_for_concat(data_format, 'WEEK')
data_format = add_spaces_for_concat(data_format, 'CALL_TYPE')
data_format = add_spaces_for_concat(data_format, 'TAXI_ID')
data_format = add_spaces_for_concat(data_format, 'DAY')



# la colonne CONTEXT_INPUT sera la concaténation du jour de la semaine, de l'heure et de la semaien de l'année pui de la colonne CALL_TYPE, de la colonne TAXI_ID, d'un espace et du dernier token de la colonne Tokenization
data_format['CONTEXT_INPUT'] =data_format['Tokenization_2'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']
#on récupère le nombre d'informations dans la colonne CONTEXT_INPUT
#Comme cette colonne contiient les informations en string séparé par un espace, on récupère la liste correspondante puis on compte le nombre d'éléments de cette liste
len_context_info = len(data_format['CONTEXT_INPUT'][0].split(' '))


def rows_attribution_cat(dataframe, nb_categories):
    nb_rows_per_cat = len(dataframe) // nb_categories
    nb_rows_dict = {f'nb_rows_category{i}': nb_rows_per_cat for i in range(nb_categories)}
    remainder = len(dataframe) % nb_categories
    
    # Distribute the remaining rows randomly among categories
    for i in random.sample(range(nb_categories), remainder):
        nb_rows_dict[f'nb_rows_category{i}'] += 1
    
    return nb_rows_dict

def create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict):
    df_dict = {f'dataframe_category{i}': dataframe.iloc[list_index_dict[f'list_index_category{i}']].copy() for i in range(nb_categories)}
    for i in range(nb_categories):
        df_dict[f'dataframe_category{i}'] = df_dict[f'dataframe_category{i}'].reindex(columns=dataframe.columns)
    return df_dict

def create_target_deb_traj(nb_categories, df_dict):
    target_dict = {f'list_target_category{i}': np.empty(len(df_dict[f'dataframe_category{i}']), dtype=object) for i in range(nb_categories)}
    list_deb_traj_dict = {f'list_deb_traj_category{i}': [[] for _ in range(len(df_dict[f'dataframe_category{i}']))] for i in range(nb_categories)}
    return target_dict, list_deb_traj_dict

def fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict):
    for i in range(nb_categories-2):
        df = df_dict[f'dataframe_category{i}']
        for j in range(len(df)):
            tokenization_2 = df.iloc[j]['Tokenization_2']
            start_idx = int(list_threshold[i] * len(tokenization_2))
            end_idx = int(list_threshold[i+1] * len(tokenization_2))
            tokenization_2 = tokenization_2[start_idx:end_idx]
            
            if len(tokenization_2) != 0:
                index = random.randint(0, len(tokenization_2)-1)
            else:
                index = -1
            
            token = tokenization_2[index]
            target_dict[f'list_target_category{i}'][j] = token
            list_deb_traj_dict[f'list_deb_traj_category{i}'][j].extend(df.iloc[j]['Tokenization_2'][:start_idx])
            list_deb_traj_dict[f'list_deb_traj_category{i}'][j].extend(tokenization_2[:index])

    i = nb_categories - 2
    df = df_dict[f'dataframe_category{i}']
    for j in range(len(df)):
        tokenization_2 = df.iloc[j]['Tokenization_2']
        token = tokenization_2[-1]
        target_dict[f'list_target_category{i}'][j] = token
        list_deb_traj_dict[f'list_deb_traj_category{i}'][j] = df.iloc[j]['Tokenization_2'][:-1]

    i = nb_categories - 1
    df = df_dict[f'dataframe_category{i}']
    for j in range(len(df)):
        target_dict[f'list_target_category{i}'][j] = '[SEP]'
        list_deb_traj_dict[f'list_deb_traj_category{i}'][j] = df.iloc[j]['Tokenization_2']

    return target_dict, list_deb_traj_dict

def prepare_train_wo_duplicate(dataframe, nb_categories=5, decal_gauche=False, decal_droite=False, uniforme=True):
    list_threshold = [0.3 + i * ((1-0.3) / (nb_categories - 2)) for i in range(nb_categories - 1)]
    random.seed(2023)

    nb_rows_dict = rows_attribution_cat(dataframe, nb_categories)
    list_index_dict = {f'list_index_category{i}': np.array(random.sample(range(len(dataframe)), nb_rows_dict[f'nb_rows_category{i}'])) for i in range(nb_categories)}

    df_dict = create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict)
    target_dict, list_deb_traj_dict = create_target_deb_traj(nb_categories, df_dict)
    target_dict, list_deb_traj_dict = fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict)

    for i in range(nb_categories):
        df_dict[f'dataframe_category{i}']['TARGET'] = target_dict[f'list_target_category{i}']
        df_dict[f'dataframe_category{i}']['DEB_TRAJ'] = list_deb_traj_dict[f'list_deb_traj_category{i}']
    
    return pd.concat([df_dict[f'dataframe_category{i}'] for i in range(nb_categories)], ignore_index=True)

# We call the function with the train-test split data
data_train, data_test = train_test_split(data_format, test_size=0.2, random_state=2023)
df_full = prepare_train_wo_duplicate(data_train)