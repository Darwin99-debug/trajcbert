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



#load the tokenizer from /home/daril_kw/data/tokenizer_final
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_final')
#load the dataset from home/daril_kw/data/data_with_time_info_ok.json
with open('/home/daril_kw/data/data_with_time_info_ok.json', 'r') as openfile:
    json_loaded = json.load(openfile)


data_format = pd.DataFrame(data=json_loaded)


data_format['HOUR']=data_format['HOUR'].apply(lambda x: ' '+x)
data_format['WEEK']=data_format['WEEK'].apply(lambda x: ' '+x)
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)
data_format['DAY']=data_format['DAY'].apply(lambda x: ' '+x)


# la colonne CONTEXT_INPUT sera la concaténation du jour de la semaine, de l'heure et de la semaien de l'année pui de la colonne CALL_TYPE, de la colonne TAXI_ID, d'un espace et du dernier token de la colonne Tokenization
data_format['CONTEXT_INPUT'] =data_format['Tokenization_2'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']
#on récupère le nombre d'informations dans la colonne CONTEXT_INPUT
#Comme cette colonne contiient les informations en string séparé par un espace, on récupère la liste correspondante puis on compte le nombre d'éléments de cette liste
len_context_info = len(data_format['CONTEXT_INPUT'][0].split(' '))


#we separate the dataframe into train and test
data_train, data_test = train_test_split(data_format, test_size=0.2, random_state=2023)






dataframe=data_train
nb_categories=5
#we create a list per category with the name of the category
categories = {}
for i in range(nb_categories):
    category_name = 'category' + str(i)
    categories[category_name] = []


#we create the threshold for each category knowing that they go from 0.3 to 1 (the last token is excluded)
#tow categories are reserved for the last token (the destination) and the [SEP] token so we don't take them into account
# for example, if ze have 5 categories, the uniform threshold would be (1-0.3)/(5-2) = 0.23333333333333334
#that means that the first category will concern length of trajectory from 0.3 to 0.5333333333333333, the second from 0.5333333333333333 to 0.7666666666666666 and the third from 0.7666666666666666 to 1
#we create a list of threshold
list_threshold = [0.3+i*((1-0.3)/(nb_categories-2)) for i in range(nb_categories-1)]


#we create a seed to be able to reproduce the results
random.seed(2023)
#wealculate the number of rows that will fall into each category
#we keep it in variables so that we can use it later
nb_rows_dict = {}
for i in range(nb_categories):
    nb_row_cat_name = 'nb_rows_category' + str(i)
    nb_rows_dict[nb_row_cat_name] = int(len(dataframe)/nb_categories)

#due to the conversion of int, we may have a number of rows that is not equal to the number of rows of the dataframe
# if the sum of the nb_rows_category is not equal to the number of rows of the dataframe
#we add the missing rows to one of the categories randomly
while sum([nb_rows_dict['nb_rows_category'+str(i)] for i in range(nb_categories)]) != len(dataframe):
    #we choose the category randomly
    index = random.randint(0,nb_categories-1)
    #we add one row to the category
    nb_rows_dict['nb_rows_category'+str(index)] += 1



#we create a list of index of the dataframe
list_index = [i for i in range(len(dataframe))]
#we shuffle the list of index
random.shuffle(list_index)
#we create a list of index for each category
list_index_dict = {}
for i in range(nb_categories):
    list_index_cat_name = 'list_index_category' + str(i)
    list_index_dict[list_index_cat_name] = []
#we fill the list of index for each category
for i in range(nb_categories):
    #we fill the list of index for each category
    list_index_dict['list_index_category'+str(i)] = list_index[:nb_rows_dict['nb_rows_category'+str(i)]]
    #we remove the index that we just put in the list
    list_index = list_index[nb_rows_dict['nb_rows_category'+str(i)]:]
#we create a list of dataframe for each category
df_dict = {}
for i in range(nb_categories):
    df_cat_name = 'dataframe_category' + str(i)
    df_dict[df_cat_name] = pd.DataFrame()
    #we use reindex so that the columns of the dataframes will be the same as the columns of the dataframe in parameter
    df_dict['dataframe_category'+str(i)] = df_dict['dataframe_category'+str(i)].reindex(columns=dataframe.columns)
    #we fill the list of dataframe for each category with the rows of the dataframe that correspond to the number of rows that we want for each category
    # ie nb_rows_category
    #fill the dataframes
    #the append does not exist for dataframes, we use the concat function
    df_dict['dataframe_category'+str(i)] = pd.concat([df_dict['dataframe_category'+str(i)],dataframe.iloc[list_index_dict['list_index_category'+str(i)]]],ignore_index=True)


#we create a list of targets and deb_traj for each category
target_dict = {}
list_deb_traj_dict = {}
for i in range(nb_categories):
    target_cat_name = 'list_target_category' + str(i)
    #we want one target per row of the dataframe of the category selected (identified by i)
    target_dict[target_cat_name] = [0 for j in range(len(df_dict['dataframe_category'+str(i)]))]
    #we want one list of deb_traj per row of the dataframe
    list_deb_traj_name = 'list_deb_traj_category' + str(i)
    list_deb_traj_dict[list_deb_traj_name]= [[] for j in range(len(df_dict['dataframe_category'+str(i)]))]

#we do the treatment for each category except the two last categories
for i in range(nb_categories-2):
    df = df_dict['dataframe_category'+str(i)]
    for j in range(len(df)):
        #we take the tokenization_2 column
        tokenization_2 = df.iloc[j]['Tokenization_2']
        #we use the threshold corresponding to the category
        tokenization_2 = tokenization_2[int(list_threshold[i]*len(tokenization_2)):int(list_threshold[i+1]*len(tokenization_2))]
        #we take randomly a token in the tokenization_2 if it is not empty ie if the list is not emptyof tokens is not too small to be cut in nb_categories
        if len(tokenization_2) != 0:
            index = random.randint(0,len(tokenization_2)-1)
        else:
            index = -1
        #we take the token
        token = tokenization_2[index]
        #we put the token in the list of target
        target_dict['list_target_category'+str(i)][j] = token
        #we put the token before the target in the list of deb_traj : we add the token befor the threshold associated to the category
        for k in df.iloc[j]['Tokenization_2'][:int(list_threshold[i]*len(df.iloc[j]['Tokenization_2']))]:
            list_deb_traj_dict['list_deb_traj_category'+str(i)][j].append(k)
        #we add the token from tokenization_2 before the target to the list of deb_traj
        for k in tokenization_2[:index]:
            list_deb_traj_dict['list_deb_traj_category'+str(i)][j].append(k)


i = nb_categories-2
df = df_dict['dataframe_category'+str(i)]
for j in range(len(df)):
    #we take the tokenization_2 column
    tokenization_2 = df.iloc[j]['Tokenization_2']
    #we take the last token
    token = tokenization_2[-1]
    #we put the token in the list of target
    target_dict['list_target_category'+str(i)][j] = token
    #we put the token before the target in the list of deb_traj : we add the token of the whole tokenization_2 before the target
    list_deb_traj_dict['list_deb_traj_category'+str(i)][j] = df.iloc[j]['Tokenization_2'][:-1]


for i in range(nb_categories-1):
    df_dict['dataframe_category'+str(i)]['TARGET'] = target_dict['list_target_category'+str(i)]
    df_dict['dataframe_category'+str(i)]['DEB_TRAJ'] = list_deb_traj_dict['list_deb_traj_category'+str(i)]

i=nb_categories-1
df_dict['dataframe_category'+str(i)]['TARGET'] = target_dict['list_target_category'+str(i)]
df_dict['dataframe_category'+str(i)]['DEB_TRAJ'] = list_deb_traj_dict['list_deb_traj_category'+str(i)]

#we verify that df_dict['dataframe_category'+str(i)]['TARGET'][j]==df_dict['dataframe_category'+str(i)]['Tokenization_2'][j][len(df_dict['dataframe_category'+str(i)]['DEB_TRAJ'][j])]
for i in range(nb_categories):
    for j in range(len(df_dict['dataframe_category'+str(i)])):
        if df_dict['dataframe_category'+str(i)]['TARGET'][j] != df_dict['dataframe_category'+str(i)]['Tokenization_2'][j][len(df_dict['dataframe_category'+str(i)]['DEB_TRAJ'][j])]:
            print(i)