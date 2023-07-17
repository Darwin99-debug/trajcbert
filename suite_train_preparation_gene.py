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

"""

def prepare_train(dataframe, sixty_percent=0.4, thirty_percent=0.3, ten_percent=0.10, last_prob=0.1, sep_prob=0.1 ):
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
    #random.shuffle(list_index)
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

    #we get back the full dataframe 
    dataframe_full = pd.concat([dataframe_sixty,dataframe_thirty,dataframe_ten,dataframe_last,dataframe_sep],ignore_index=True)
    # we put it in the right order (using the TRIP_ID column so that the dataframe is in the same order as the dataframe in input called dataframe)
    #dataframe_full = dataframe_full.sort_values(by=['TRIP_ID'])
    #we verify that the dataframe is in the right order
    #dataframe=dataframe.sort_values(by=['TRIP_ID'])
    for i in range(len(dataframe_full)):
        if dataframe_full['TRIP_ID'][i] != dataframe['TRIP_ID'][i]:
            raise ValueError('the dataframe is not in the right order') 
    
    return dataframe_full


df_full = prepare_train(data_train, 0.45, 0.3,0.15,0.05,0.05)
"""

def prepare_train_wo_duplicate(dataframe, nb_categories=5, decal_gauche=False, decal_droite=False, uniforme=True):
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
        target_dict[target_cat_name] = [0 for i in range(len(df_dict['dataframe_category'+str(i)]))]
        #we want one list of deb_traj per row of the dataframe
        list_deb_traj_dict[target_cat_name]= [[] for i in range(len(df_dict['dataframe_category'+str(i)]))]


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
            #we put the token before the target in the list of deb_traj : we add the token befor the first threshold fisrt
            for k in df.iloc[j]['Tokenization_2'][:int(list_threshold[i]*len(df.iloc[j]['Tokenization_2']))]:
                list_deb_traj_dict['list_deb_traj_category'+str(i)][j].append(k)
            #we add the token from tokenization_2 before the target to the list of deb_traj
            for k in tokenization_2[:index]:
                target_dict['list_deb_traj_category'+str(i)][j].append(k)


            
    #for the penultimate category, we take the last token of the tokenization_2
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

    #for the last category, we take the as target the [SEP] token
    i = nb_categories-1
    df = df_dict['dataframe_category'+str(i)]
    for j in range(len(df)):
        target_dict['list_target_category'+str(i)][j] = '[SEP]'
        #we put the token before the target in the list of deb_traj : we add the token of the whole tokenization_2
        list_deb_traj_dict['list_deb_traj_category'+str(i)][j] = df.iloc[j]['Tokenization_2']
                
    #we add the lists of target in the column target of the dataframe and same for deb_traj
    df_dict['dataframe_category'+str(i)]['TARGET'] = target_dict['list_target_category'+str(i)]
    df_dict['dataframe_category'+str(i)]['DEB_TRAJ'] = list_deb_traj_dict['list_deb_traj_category'+str(i)]

    #we verifiy that for each category exept the last one dataframe'Tokenization_2'][i][len(dataframe['DEB_TRAJ'][i])]!=dataframe['TARGET'][i]
    # wuere i goes from 0 to len(dataframe)
    #and the dataframe is the dataframe_category
    for i in range(nb_categories-1):
        df = df_dict['dataframe_category'+str(i)]
        for j in range(len(df)):
            if df['Tokenization_2'][j][len(df['DEB_TRAJ'][j])]!=df['TARGET'][j]:
                print(i,j)

    #we get the full dataframe 
    dataframe_full=pd.concat([df_dict['dataframe_category'+str(i)] for i in range(nb_categories)],ignore_index=True)
    #we put it in the right order
    dataframe_full = dataframe_full.reindex(columns=dataframe.columns)

    return dataframe_full

#we call the function
df_full = prepare_train_wo_duplicate(data_train)

            



    


