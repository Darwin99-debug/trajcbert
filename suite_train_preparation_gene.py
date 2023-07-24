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


#we separate the dataframe into train and test 
data_train, data_test = train_test_split(data_format, test_size=0.2, random_state=2023)



def rows_attribution_cat(dataframe, nb_categories):
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

    return nb_rows_dict, list_index_dict



def create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict):
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
    return df_dict


def create_target_deb_traj(nb_categories, df_dict):
    target_dict = {}
    list_deb_traj_dict = {}
    for i in range(nb_categories):
        target_cat_name = 'list_target_category' + str(i)
        #we want one target per row of the dataframe of the category selected (identified by i)
        target_dict[target_cat_name] = [0 for j in range(len(df_dict['dataframe_category'+str(i)]))]
        #we want one list of deb_traj per row of the dataframe
        list_deb_traj_name = 'list_deb_traj_category' + str(i)
        list_deb_traj_dict[list_deb_traj_name]= [[] for j in range(len(df_dict['dataframe_category'+str(i)]))]
    return target_dict, list_deb_traj_dict

def fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict):
    
    
    
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

    return target_dict, list_deb_traj_dict

def verif_target_deb_traj(df_dict, nb_categories):
    for i in range(nb_categories-1):
        df = df_dict['dataframe_category'+str(i)]
        for j in range(len(df)):
            if df['Tokenization_2'][j][len(df['DEB_TRAJ'][j])]!=df['TARGET'][j]:
                #raise ValueError('The target is not the next token of the deb_traj')
                raise ValueError('The target is not the next token of the deb_traj')

def prepare_train_wo_duplicate(dataframe, nb_categories=5, liste_to_duplicate=[], decal_gauche=False, decal_droite=False, uniforme=True):
    """
liste_to_duplicate is a list of TAXI_ID that we want to duplicate """
    #we create the threshold for each category knowing that they go from 0.3 to 1 (the last token is excluded)
    #tow categories are reserved for the last token (the destination) and the [SEP] token so we don't take them into account
    # for example, if ze have 5 categories, the uniform threshold would be (1-0.3)/(5-2) = 0.23333333333333334
    #that means that the first category will concern length of trajectory from 0.3 to 0.5333333333333333, the second from 0.5333333333333333 to 0.7666666666666666 and the third from 0.7666666666666666 to 1
    #we create a list of threshold
    list_threshold = [0.3+i*((1-0.3)/(nb_categories-2)) for i in range(nb_categories-1)]

    #we remove the useless rows (those wo have an element that is not a list in the column Tokenization_2)
    dataframe['Tokenization_2']=dataframe['Tokenization_2'].apply(lambda x: x if type(x)==list else [])
    #we remove from the dataframe the rows that have a trajectory of length inferior to 3
    dataframe['LEN_TRAJ']=dataframe['Tokenization_2'].apply(lambda x: len(x))
    dataframe = dataframe[dataframe['LEN_TRAJ']>=3]

    for i in range(len(liste_to_duplicate)):
        #we wont enter the loop if the list is empty
        #we add the rows to duplicate to the dataframe
        dataframe = pd.concat([dataframe,dataframe[dataframe['TAXI_ID']==liste_to_duplicate[i]]],ignore_index=True)


    #we create a seed to be able to reproduce the results
    random.seed(2023)
    #wealculate the number of rows that will fall into each category
    #we keep it in variables so that we can use it later
    nb_rows_dict, list_index_dict = rows_attribution_cat(dataframe, nb_categories)
    


    #we create a list of dataframe for each category 
    df_dict = create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict)

    #we create a list of targets and deb_traj for each category
    target_dict, list_deb_traj_dict = create_target_deb_traj(nb_categories, df_dict)    
    target_dict, list_deb_traj_dict = fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict)


    #we add the lists of target in the column target of the dataframe and same for deb_traj
    for i in range(nb_categories):
        df_dict['dataframe_category'+str(i)]['TARGET'] = target_dict['list_target_category'+str(i)]
        df_dict['dataframe_category'+str(i)]['DEB_TRAJ'] = list_deb_traj_dict['list_deb_traj_category'+str(i)]
    

    #we verify that for each category exept the last one dataframe'Tokenization_2'][i][len(dataframe['DEB_TRAJ'][i])]!=dataframe['TARGET'][i]
    # wuere i goes from 0 to len(dataframe)
    #and the dataframe is the dataframe_category           
    verif_target_deb_traj(df_dict, nb_categories)
    
    #we get the full dataframe back
    dataframe_full = pd.DataFrame()
    for i in range(nb_categories):
        dataframe_full = pd.concat([dataframe_full,df_dict['dataframe_category'+str(i)]],ignore_index=True)

    return dataframe_full

#we call the function
df_full = prepare_train_wo_duplicate(data_train)


def manage_separation_test(dataframe, list_index_to_separate):
    #we manage the separation 
    dict_row = {}
    dataframe_separated = dataframe.copy()
    #we reinitalize the index of the dataframe
    dataframe_separated.reset_index(drop=True, inplace=True)


    #we track the rows thanks to the TRIP_ID and put their index in a list
    list_index = [j for j in range(len(dataframe)) if dataframe['TRIP_ID'][j] in [list_index_to_separate[i][0] for i in range(len(list_index_to_separate))]]



    for i in range(len(list_index_to_separate)):
        #we select the row in data_train thanks to the TRIP_ID
        #row = dataframe_separated[data_train['TRIP_ID']==list_index_to_separate[i][0]]
        #if the above line does not work, we can use the following line
        #row = dataframe_separated.iloc[list_index_to_separate[i][0]]
        #this line works only if the index of the dataframe is the same as the TRIP_ID but it is not the case so we have to transform the TRIP_ID into the index
        
        row = dataframe.iloc[list_index[i]]


        #row contains the row that we will separate
        #we remove the row from the dataframe and replace it by the same row but with the Tokenization_2 column that is a piece of the Tokenization_2 column of the row seperated in list_index_to_separate[i][1] trajectories
        #we remove the original row from the dataframe but we keep it in the variable row
    
        dataframe_separated = dataframe.drop(list_index[i], axis=0)
        #we create the list of trajectories
        list_traj = []
        #WE FILL THE LIST OF TRAJECTORIES
        #we take the Tokenization_2 column
        tokenization_2 = row.iloc[0]['Tokenization_2']
        #we take the length of the trajectory
        len_traj = len(tokenization_2)
        #we take the number of trajectories
        nb_traj = list_index_to_separate[i][1]
        #we take the length of each trajectory
        len_each_traj = len_traj//nb_traj
        #we fill the list of trajectories
        for j in range(nb_traj):
            #we take the piece of the trajectory
            traj = tokenization_2[j*len_each_traj:(j+1)*len_each_traj]
            #we put it in the list of trajectories
            list_traj.append(traj)
        #if there is a rest, we add it to the last trajectory
        rest = len_traj%nb_traj
        if rest != 0:
            list_traj[-1].append(tokenization_2[-rest:])
        #we add the trajectories to the dataframe in new rows
        for j in range(nb_traj):
            #we create a new row that will be added to the dataframe, for that we can use the function  concat
            dataframe_separated = pd.concat([dataframe_separated,row],ignore_index=True)
            #we add the trajectory to the Tokenization_2 column
            dataframe_separated.loc[[len(dataframe_separated)-1],['Tokenization_2']] = list_traj[j]
        
    return dataframe_separated



def manage_separation(dataframe, list_index_to_separate):
    dataframe_separated = dataframe.copy()
    dataframe_separated.reset_index(drop=True, inplace=True)

    # Create a list of TRIP_IDs from list_index_to_separate
    trip_ids_to_separate = [item[0] for item in list_index_to_separate]

    # Sort both list_index and list_index_to_separate based on the TRIP_ID values
    list_index = [idx for idx in dataframe_separated.index if dataframe_separated.loc[idx, 'TRIP_ID'] in trip_ids_to_separate]
    list_index_to_separate = sorted(list_index_to_separate, key=lambda x: x[0])

    modified_rows = []

    # Rest of the code remains the same
    for i in range(len(list_index_to_separate)):
        row = dataframe_separated.loc[list_index[i]].copy()
        dataframe_separated.drop(list_index[i], inplace=True)

        list_traj = []
        tokenization_2 = row['Tokenization_2']
        len_traj = len(tokenization_2)
        nb_traj = list_index_to_separate[i][1]
        len_each_traj = len_traj // nb_traj
        for j in range(nb_traj):
            traj = tokenization_2[j * len_each_traj:(j + 1) * len_each_traj]
            list_traj.append(traj)
        rest = len_traj % nb_traj
        if rest != 0:
            list_traj[-1].append(tokenization_2[-rest:])

        for j in range(nb_traj):
            new_row = row.copy()
            new_row['Tokenization_2'] = list_traj[j]
            modified_rows.append(new_row)

    # Concatenate all the modified rows in one operation
    dataframe_separated = pd.concat([dataframe_separated] + modified_rows, ignore_index=True)

    return dataframe_separated

def prepare_train(dataframe, duplication_rate=0, separation_rate=50):
    """
    This function prepares the train dataset like the prepare_train_wo_duplicate function but with the possibility to duplicate the rows.
    The separation rate is the proportion of rows that will separated into two different trajectories. 
    The duplication rate is the proportion of rows that will be duplicated, ie that will occur in two different trajectories with different targets.

    """
    #we select the rows we are going to separate or duplicate
    nb_to_separate = int(len(dataframe)*separation_rate/100)
    nb_to_duplicate = int(len(dataframe)*duplication_rate/100)
    nb_to_select=nb_to_separate+nb_to_duplicate

    #we create a seed to be able to reproduce the results
    random.seed(2023)
    #we select the rows we are going to separate or duplicate according to their length : we select the rows with the longest trajectories
    #for that, we sort the dataframe by the length of the trajectory
    #the thing is, the two part of a trajectory can be longer than the following longest trajectory
    #that means we can have the longest trajectory of length 512 for example, wich means the two resulting trajectories will be of length 256
    #but the 2nd longest trajectory is of length 255, so the two resulting trajectories will be longer than the 2nd longest trajectory
    # that is why we need to sort at each iteration
    #we create a dataframe that zill be the fisrt dataframe but sorted by the length of the trajectory and we keep the matching before sorting to ne able to find the rows in the original dataframe
    dataframe["LEN_TRAJ"]=dataframe['Tokenization_2'].apply(lambda x: len(x))
    sorted_dataframe= dataframe.sort_values(by=['LEN_TRAJ'], ascending=False)
    #we will track he rows thaks to the TRIP_ID
    list_row_to_select = [ [] for i in range(nb_to_select)]
    for i in range(nb_to_select):
        j=2
        while len(sorted_dataframe.iloc[i]['Tokenization_2'])//j>len(sorted_dataframe.iloc[i+1]['Tokenization_2']):
            j+=1
        #j represents the number of trajectories that we will create from the trajectory i, we put it in a list
        list_row_to_select[i].append(sorted_dataframe.iloc[i]['TRIP_ID'])
        list_row_to_select[i].append(j)

    #now that we have the rows that we will separate or duplicate, we can select which rows we will separate and which we will duplicate
    #we create a list of index of the rows that we will separate
    list_index_to_separate = []
    #we create a list of index of the rows that we will duplicate
    list_index_to_duplicate = []
    #we choose randomly the rows that we will separate and the rows that we will duplicate
    list_index_to_separate = random.sample(list_row_to_select, nb_to_separate)
    #we take the rows that we did not select for separation
    list_index_to_duplicate = [i for i in list_row_to_select if i not in list_index_to_separate]

    dataframe_separated=manage_separation(dataframe, list_index_to_separate)

    #we call the funtion prepare_train_wo_duplicate with the list of rows to duplicate
    df_full = prepare_train_wo_duplicate(dataframe_separated, liste_to_duplicate=list_index_to_duplicate)

    return df_full, dataframe_separated, list_index_to_separate

#we call the function
df_full2, df_sep, list_row_to_sep = prepare_train(data_train, duplication_rate=0, separation_rate=50)





#after that, we verify that the rows that we separated are well separated
#we us the list_row_to_sep that we created in the prepare_train function to see if the rows that we separated are well separated
# for j that goes from 0 to len(list_row_to_sep), we verify that the number of rows with the trip_id list_row_to_sep[j][0] is equal to list_row_to_sep[j][1]

def verif_separation(dataframe, list_row_to_sep):
    for j in range(len(list_row_to_sep)):
        if len(dataframe[dataframe['TRIP_ID']==list_row_to_sep[j][0]])!=list_row_to_sep[j][1]:
            print(j, list_row_to_sep[j][0], list_row_to_sep[j][1], len(dataframe[dataframe['TRIP_ID']==list_row_to_sep[j][0]]))
    return 'The rows are well separated'

verif_separation(df_sep, list_row_to_sep)




    


