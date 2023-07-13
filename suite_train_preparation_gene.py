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



def prepare_train(dataframe, sixty_percent=0.4, thirty_percent=0.25, ten_percent=0.15, last_prob=0.1, sep_prob=0.1 ):
    #sixty_percent is the proportion of the time we will take the target between the 60% last tokens and the 30% last tokens
    #thirty_percent is the proportion of the time we will take the target between the 30% last tokens and the 10% last tokens
    #ten_percent is the proportion of the time we will take the target in the 10% last tokens
    #last_prob is the proportion of the time we will take the very last token as target
    #sep_prob is the proportion of the time we will take the [SEP] token (after the very last token) as target
    #the sum of the five parameters must be equal to 1, we verify it
    #if it is not the case, we raise an error
    if sixty_percent+thirty_percent+ten_percent+last_prob+sep_prob!=1:
        raise ValueError('The sum of the five parameters must be equal to 1')
    #random.random() return a random float number between 0 and 1
    
    #we create a list of the rows that will fall into the 5 categories (between 60% and 30%, between 30% and 10%, in the 10%, the very last token and the [SEP] token)
    list_sixty = []
    list_thirty = []
    list_ten = []
    list_last = []
    list_sep = []
    #we create a seed to be able to reproduce the results
    random.seed(2023)
    #wealculate the number of rows that will fall into each category
    number_sixty = int(len(dataframe)*sixty_percent)
    number_thirty = int(len(dataframe)*thirty_percent)
    number_ten = int(len(dataframe)*ten_percent)
    number_last = int(len(dataframe)*last_prob)
    number_sep = int(len(dataframe)*sep_prob)

    #due to the conversion of int, we may have a number of rows that is not equal to the number of rows of the dataframe
    # if the sum of number_sixty, number_thirty, number_ten, number_last and number_sep is not equal to the number of rows of the dataframe
    #we add the missing rows to one of the categories randomly
    while number_sixty+number_thirty+number_ten+number_last+number_sep!=len(dataframe):
        #we choose randomly one of the categories
        category = random.choice(['sixty', 'thirty', 'ten', 'last', 'sep'])
        #we add one row to the category
        if category == 'sixty':
            number_sixty+=1
        elif category == 'thirty':
            number_thirty+=1
        elif category == 'ten':
            number_ten+=1
        elif category == 'last':
            number_last+=1
        else :
            number_sep+=1



    #we create a list of the indexes of the rows of the dataframe
    list_index = list(range(len(dataframe)))
    #we shuffle the list of indexes
    random.shuffle(list_index)
    #we create a list of the indexes of the rows that will fall into each category
    list_index_sixty = list_index[:number_sixty]
    list_index_thirty = list_index[number_sixty:number_sixty+number_thirty]
    list_index_ten = list_index[number_sixty+number_thirty:number_sixty+number_thirty+number_ten]
    list_index_last = list_index[number_sixty+number_thirty+number_ten:number_sixty+number_thirty+number_ten+number_last]
    list_index_sep = list_index[number_sixty+number_thirty+number_ten+number_last:number_sixty+number_thirty+number_ten+number_last+number_sep]
    
    #we create 5 dataframes of the rows that will fall into the 5 categories (between 60% and 30%, between 30% and 10%, in the 10%, the very last token and the [SEP] token)
    #declare the dataframes
    dataframe_sixty = pd.DataFrame()
    dataframe_thirty = pd.DataFrame()
    dataframe_ten = pd.DataFrame()
    dataframe_last = pd.DataFrame()
    dataframe_sep = pd.DataFrame()
    #the columns of the dataframes will be the same as the columns of the dataframe in parameter
    dataframe_sixty = dataframe_sixty.reindex(columns=dataframe.columns)
    dataframe_thirty = dataframe_thirty.reindex(columns=dataframe.columns)
    dataframe_ten = dataframe_ten.reindex(columns=dataframe.columns)
    dataframe_last = dataframe_last.reindex(columns=dataframe.columns)
    dataframe_sep = dataframe_sep.reindex(columns=dataframe.columns)
    #fill the dataframes 
    #the append does not exist for dataframes, we use the concat function 
    dataframe_sixty = pd.concat([dataframe_sixty, dataframe.iloc[list_index_sixty]])
    dataframe_thirty = pd.concat([dataframe_thirty, dataframe.iloc[list_index_thirty]])
    dataframe_ten = pd.concat([dataframe_ten, dataframe.iloc[list_index_ten]])
    dataframe_last = pd.concat([dataframe_last, dataframe.iloc[list_index_last]])
    dataframe_sep = pd.concat([dataframe_sep, dataframe.iloc[list_index_sep]])


    #for each dataframe, we take the tokenization_2 column and we choose the target according to the category

    #to avoid the caveat of the dataframe, we use the copy function
    dataframe_sixty = dataframe_sixty.copy()
    dataframe_thirty = dataframe_thirty.copy()
    dataframe_ten = dataframe_ten.copy()
    dataframe_last = dataframe_last.copy()
    dataframe_sep = dataframe_sep.copy()



    #for the dataframe_sixty

    # we create a list of targets
    list_target_full_data = [0 for i in range(len(dataframe))]
    list_target_sixty_data = [0 for i in range(len(dataframe_sixty))]
    list_deb_traj_full_data = [0 for i in range(len(dataframe))]
    list_deb_traj_sixty_data = [0 for i in range(len(dataframe_sixty))]
    for i in range(len(dataframe_sixty)):
        
        #we take the tokenization_2 column
        tokenization_2 = dataframe_sixty.iloc[i]['Tokenization_2']
        #we take the token that is between the 60% and the 30% last tokens
        tokenization_2 = tokenization_2[-int(len(tokenization_2)*0.6):-int(len(tokenization_2)*0.3)]
        #we take a random index in the list
        if len(tokenization_2) != 0:
            index = random.randint(0,len(tokenization_2)-1)
        else:
            index = -1
        #list_target_full_data[list_index_sixty[i]] = tokenization_2[index]
        list_target_sixty_data[i] = tokenization_2[index]
        #we put in the column DEB_TRAJ the tokens that are before the target
        #we take as deb_traj the tokens that are before the target ie the tokens that are before the 60% last tokens and those that are between the 60% and the 30% last tokens and at the sqme time before the index
        #we begin by taking the tokens that are before the 60% last tokens 
        list_deb_traj_sixty_data[i] = dataframe_sixty.iloc[i]['Tokenization_2'][:-int(len(dataframe_sixty.iloc[i]['Tokenization_2'])*0.6)]
        #then we add the tokens that are between the 60% and the 30% last tokens and before the index
        if index != 0:
            #we append the element of tokenization_2[:index] to the list and not the list in itself
            for token in tokenization_2[:index] :
                list_deb_traj_sixty_data[i].append(token)
        
        
        
        """
        tokenization_2 = dataframe_sixty.iloc[i]['Tokenization_2']
        #we take the token that is between the 60% and the 30% last tokens
        tokenization_2 = tokenization_2[-int(len(tokenization_2)*0.6):-int(len(tokenization_2)*0.3)]
        #we take a random index in the list
        index = random.randint(0,len(tokenization_2)-1)
        dataframe_sixty.iloc[i]['TARGET'] = tokenization_2[index]
        dataframe.iloc[list_index_sixty[i]]['TARGET'] = tokenization_2[index]
        #we put in the column DEB_TRAJ the tokens that are before the target
        dataframe_sixty.iloc[i]['DEB_TRAJ'] =  dataframe_sixty.iloc[i]['Tokenization_2'][:-int(len(tokenization_2)*0.6)+index]
        dataframe.iloc[list_index_sixty[i]]['DEB_TRAJ'] =  dataframe.iloc[list_index_sixty[i]]['Tokenization_2'][:-int(len(tokenization_2)*0.6)+index]
        """




        """
        #we put the token in the target column



        dataframe_sixty.iloc[i]['TARGET'] = tokenization_2[index]
        dataframe.iloc[list_index_sixty[i]]['TARGET'] = tokenization_2[index]
        #we put in the column DEB_TRAJ the tokens that are before the target
        dataframe_sixty.iloc[i]['DEB_TRAJ'] =  dataframe_sixty.iloc[i]['Tokenization_2'][:-int(len(tokenization_2)*0.6)+index]
        dataframe.iloc[list_index_sixty[i]]['DEB_TRAJ'] =  dataframe.iloc[list_index_sixty[i]]['Tokenization_2'][:-int(len(tokenization_2)*0.6)+index]
        """
    dataframe_sixty['TARGET'] = list_target_sixty_data
    
    dataframe_sixty['DEB_TRAJ'] = list_deb_traj_sixty_data
   
    # we verify that dataframe_sixty['Tokenization_2'][i][len(dataframe_sixty['DEB_TRAJ'][i])]==dataframe_sixty['TARGET'][i] for all i
    #this would mean that the target is the token that is after the deb_traj
    for i in range(len(dataframe_sixty)):
        if dataframe_sixty['Tokenization_2'][i][len(dataframe_sixty['DEB_TRAJ'][i])]!=dataframe_sixty['TARGET'][i]:
            #raise ValueError('the target is not the token that is after the deb_traj') :
            raise ValueError('the target is not the token that is after the deb_traj')



    #for the dataframe_thirty
    list_target_thirty_data = [0 for i in range(len(dataframe_thirty))]
    list_deb_traj_thirty_data = [0 for i in range(len(dataframe_thirty))]

    for i in range(len(dataframe_thirty)):
        tokenization_2 = dataframe_thirty.iloc[i]['Tokenization_2']
        tokenization_2 = tokenization_2[-int(len(tokenization_2)*0.3):-int(len(tokenization_2)*0.1)]
        if len(tokenization_2) != 0:
            index = random.randint(0,len(tokenization_2)-1)
        else:
            index = -1
        #list_target_full_data[list_index_thirty[i]] = tokenization_2[index]
        list_target_thirty_data[i] = tokenization_2[index]
        #list_deb_traj_full_data[list_index_thirty[i]] = dataframe.iloc[list_index_thirty[i]]['Tokenization_2'][:-int(len(tokenization_2)*0.3)+index]
        list_deb_traj_thirty_data[i] = dataframe_thirty.iloc[i]['Tokenization_2'][:-int(len(dataframe_thirty.iloc[i]['Tokenization_2'])*0.3)]
        #then we add the tokens that are between the 30% and the 10% last tokens and before the index
        if index != 0:
            #we append the element of tokenization_2[:index] to the list and not the list in itself
            for token in tokenization_2[:index] :
                list_deb_traj_thirty_data[i].append(token)
        

    dataframe_thirty['TARGET'] = list_target_thirty_data
    dataframe_thirty['DEB_TRAJ'] = list_deb_traj_thirty_data

    for i in range(len(dataframe_thirty)):
        if dataframe_thirty['Tokenization_2'][i][len(dataframe_thirty['DEB_TRAJ'][i])]!=dataframe_thirty['TARGET'][i]:
            #raise ValueError('the target is not the token that is after the deb_traj') :
            raise ValueError('the target is not the token that is after the deb_traj')

    #for the dataframe_ten

    list_target_ten_data = [0 for i in range(len(dataframe_ten))]
    list_deb_traj_ten_data = [0 for i in range(len(dataframe_ten))]

    for i in range(len(dataframe_ten)):
        tokenization_2 = dataframe_ten.iloc[i]['Tokenization_2']
        tokenization_2 = tokenization_2[-int(len(tokenization_2)*0.1):-1]
        if len(tokenization_2) != 0:
            index = random.randint(0,len(tokenization_2)-1)
            list_target_ten_data[i] = tokenization_2[index]
        #list_deb_traj_full_data[list_index_ten[i]] = dataframe.iloc[list_index_ten[i]]['Tokenization_2'][:-int(len(tokenization_2)*0.1)+index]
            list_deb_traj_ten_data[i] = dataframe_ten.iloc[i]['Tokenization_2'][:-int(len(dataframe_ten.iloc[i]['Tokenization_2'])*0.1)]
        #then we add the tokens that are between the 10% and the last token and before the index
            if index != 0:
            #we append the element of tokenization_2[:index] to the list and not the list in itself
                for token in tokenization_2[:index] :
                    list_deb_traj_ten_data[i].append(token)
        else:
            list_target_ten_data[i] = dataframe_ten.iloc[i]['Tokenization_2'][-1]
            list_deb_traj_ten_data[i] = dataframe_ten.iloc[i]['Tokenization_2'][:-1]
        

    dataframe_ten['TARGET'] = list_target_ten_data
    dataframe_ten['DEB_TRAJ'] = list_deb_traj_ten_data

    for i in range(len(dataframe_ten)):
        if dataframe_ten['Tokenization_2'][i][len(dataframe_ten['DEB_TRAJ'][i])]!=dataframe_ten['TARGET'][i]:
            #raise ValueError('the target is not the token that is after the deb_traj') :
            raise ValueError('the target is not the token that is after the deb_traj')


        

    #for the dataframe_last
    list_target_last_data = [0 for i in range(len(dataframe_last))]
    list_deb_traj_last_data = [0 for i in range(len(dataframe_last))]
    for i in range(len(dataframe_last)):
        #list_target_full_data[list_index_last[i]] = tokenization_2
        list_target_last_data[i] = dataframe_last.iloc[i]['Tokenization_2'][-1]
        #list_deb_traj_full_data[list_index_last[i]] = dataframe.iloc[list_index_last[i]]['Tokenization_2'][:-1]
        list_deb_traj_last_data[i] = dataframe_last.iloc[i]['Tokenization_2'][:-1]

    dataframe_last['TARGET'] = list_target_last_data
    dataframe_last['DEB_TRAJ'] = list_deb_traj_last_data

    #for the dataframe_sep
    list_target_sep_data = [0 for i in range(len(dataframe_sep))]
    list_deb_traj_sep_data = [0 for i in range(len(dataframe_sep))]
    for i in range(len(dataframe_sep)):
       #list_target_full_data[list_index_sep[i]] = '[SEP]'
        list_target_sep_data[i] = '[SEP]'
        #list_deb_traj_full_data[list_index_sep[i]] = dataframe.iloc[list_index_sep[i]]['Tokenization_2']
        list_deb_traj_sep_data[i] = dataframe_sep.iloc[i]['Tokenization_2']

    dataframe_sep['TARGET'] = list_target_sep_data
    dataframe_sep['DEB_TRAJ'] = list_deb_traj_sep_data

    #we get back the full dataframe IN THE ORIGINAL ORDER
    dataframe_full = pd.concat([dataframe_sixty,dataframe_thirty,dataframe_ten,dataframe_last,dataframe_sep],ignore_index=True)
    return dataframe_full

df_full = prepare_train(data_train, 0.4, 0.25,0.15,0.1,0.1)