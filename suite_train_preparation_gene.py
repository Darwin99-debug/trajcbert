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
liste_to_duplicate is a list of TRIP_ID that we want to duplicate 
    we create the threshold for each category knowing that they go from 0.3 to 1 (the last token is excluded)
    tow categories are reserved for the last token (the destination) and the [SEP] token so we don't take them into account
    for example, if ze have 5 categories, the uniform threshold would be (1-0.3)/(5-2) = 0.23333333333333334
    that means that the first category will concern length of trajectory from 0.3 to 0.5333333333333333, the second from 0.5333333333333333 to 0.7666666666666666 and the third from 0.7666666666666666 to 1
    we create a list of threshold"""

    # Create the threshold for each category
    # Create the threshold for each category
    list_threshold = [0.3 + i * ((1 - 0.3) / (nb_categories - 2)) for i in range(nb_categories - 1)]

    # Remove the useless rows and rows with trajectory length < 3
    dataframe['Tokenization_2'] = dataframe['Tokenization_2'].apply(lambda x: x if type(x) == list else [])
    dataframe['LEN_TRAJ'] = dataframe['Tokenization_2'].apply(lambda x: len(x))
    dataframe = dataframe[dataframe['LEN_TRAJ'] >= 3]

    # Convert liste_to_duplicate elements to tuples and create a set
    liste_to_duplicate = set(tuple(map(np.int64, [item[0]])) for item in liste_to_duplicate)

    # Create a list to store duplicated DataFrames
    duplicated_dfs = []

    # Duplicate rows for each unique TRIP_ID value
    for trip_id in liste_to_duplicate:
        # Convert trip_id back to a list before comparison
        trip_id_list = list(trip_id)
        # Filter rows with the current trip_id and add them to the list of duplicated DataFrames
        rows_to_append = dataframe[dataframe['TRIP_ID'].apply(lambda x: np.array_equal(x, trip_id_list))]
        duplicated_dfs.append(rows_to_append)

    # Concatenate all duplicated DataFrames
    if len(liste_to_duplicate) > 0:
        duplicated_rows = pd.concat(duplicated_dfs, ignore_index=True)

    # Create a seed to be able to reproduce the results
    random.seed(2023)
    # Calculate the number of rows that will fall into each category
    nb_rows_dict, list_index_dict = rows_attribution_cat(dataframe, nb_categories)

    # Create a list of dataframe for each category
    df_dict = create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict)

    # Create a list of targets and deb_traj for each category
    target_dict, list_deb_traj_dict = create_target_deb_traj(nb_categories, df_dict)
    target_dict, list_deb_traj_dict = fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict)

    # Add the lists of target and deb_traj in the dataframe for each category
    for i in range(nb_categories):
        df_dict['dataframe_category' + str(i)]['TARGET'] = target_dict['list_target_category' + str(i)]
        df_dict['dataframe_category' + str(i)]['DEB_TRAJ'] = list_deb_traj_dict['list_deb_traj_category' + str(i)]

    # Verify that for each category except the last one dataframe['Tokenization_2'][i][len(dataframe['DEB_TRAJ'][i])] != dataframe['TARGET'][i]
    verif_target_deb_traj(df_dict, nb_categories)

    # Get the full dataframe back
    dataframe_full = pd.DataFrame()
    for i in range(nb_categories):
        dataframe_full = pd.concat([dataframe_full, df_dict['dataframe_category' + str(i)]], ignore_index=True)

    return dataframe_full

#we call the function
df_full = prepare_train_wo_duplicate(data_train)




def manage_separation(dataframe, list_index_to_separate):
    dataframe_separated = dataframe.copy()
    dataframe_separated.reset_index(drop=True, inplace=True)

    # Create a dictionary to map TRIP_ID to its index in list_index_to_separate
    trip_id_to_index = {trip_id: i for i, (trip_id, _) in enumerate(list_index_to_separate)}

    # Sort list_index based on the order of TRIP_ID in list_index_to_separate
    list_index = sorted(dataframe_separated.index, key=lambda idx: trip_id_to_index.get(dataframe_separated.loc[idx, 'TRIP_ID'], float('inf')))

    modified_rows = []

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

    for i in range(len(list_index_to_separate)):
        idx_to_remove = [idx for idx in dataframe_separated.index if dataframe_separated.loc[idx, 'TRIP_ID'] == list_index_to_separate[i][0]]
        dataframe_separated.drop(idx_to_remove, inplace=True)

    for row in modified_rows:
        dataframe_separated = pd.concat([dataframe_separated, row.to_frame().T])

    dataframe_separated.reset_index(drop=True, inplace=True)

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

    return df_full, dataframe_separated, list_index_to_separate, list_index_to_duplicate

#we call the function
df_full2, df_sep, list_row_to_sep, unused = prepare_train(data_train, duplication_rate=0, separation_rate=50)





#after that, we verify that the rows that we separated are well separated
#we us the list_row_to_sep that we created in the prepare_train function to see if the rows that we separated are well separated
# for j that goes from 0 to len(list_row_to_sep), we verify that the number of rows with the trip_id list_row_to_sep[j][0] is equal to list_row_to_sep[j][1]

def verif_separation(dataframe, list_row_to_sep):
    for j in range(len(list_row_to_sep)):
        if len(dataframe[dataframe['TRIP_ID']==list_row_to_sep[j][0]])!=list_row_to_sep[j][1]:
            raise ValueError('The rows are not well separated')
    return 'The rows are well separated'

verif_separation(df_sep, list_row_to_sep)

#part of the verification is to see wheter the concatenation of the trajectories is equal to the original trajectory
#for that we use df_full and df_sep and see if the concatenation of Tokenization_2 of df_sep is equal to the Tokenization_2 of df_full
#for that, we count whether the number of points of the concatenation is equal to the number of points of the original trajectory (column Tokenization_2 of df_full)
def verif_concatenation(df_full, df_sep):
    for i in range(len(df_full)):
        #we get the rows that have the same TRIP_ID in df_sep as the row i of df_full
        df = df_sep[df_sep['TRIP_ID']==df_full['TRIP_ID'][i]]
        #we get the sum of the length of the trajectories of df
        sum_len_traj = sum([len(df.iloc[j]['Tokenization_2']) for j in range(len(df))])
        #we get the length of the original trajectory
        len_traj = len(df_full['Tokenization_2'][i])
        #we verify that the sum of the length of the trajectories of df is equal to the length of the original trajectory
        if sum_len_traj != len_traj:
            raise ValueError('The concatenation of the trajectories is not equal to the original trajectory') 
    return 'The concatenation of the trajectories is equal to the original trajectory'


a=verif_concatenation(df_full, df_sep)





    
#on pase à la duplication
df_full_dup, df_sep_dup, list_row_to_sep_dup, list_row_to_dup = prepare_train(data_train, duplication_rate=30, separation_rate=50)
df_full_dup1, df_sep_dup1, list_row_to_sep_dup1, list_row_to_dup1 = prepare_train(data_train, duplication_rate=50, separation_rate=0)
#on verifie la longueur du dataframe df_full_dup qui doit être égale à la longueur du dataframe df_full + le nombre de lignes dupliquées + le nombre de lignes séparées (selon en combien de traj on a séparé les lignes)
nb_lignes_sep1=0
for i in range(len(list_row_to_sep_dup1)):
    nb_lignes_sep1+=list_row_to_sep_dup1[i][1]
if len(df_full_dup1) != len(df_full) + len(list_row_to_dup1) + nb_lignes_sep1 - len(list_row_to_sep_dup1):
    print(len(df_full_dup1), len(df_full) + len(list_row_to_dup1) + nb_lignes_sep1 - len(list_row_to_sep_dup1))


#we get the lines that must have been duplicated
df_dup = df_full_dup[df_full_dup['TRIP_ID'].isin([list_row_to_dup1[i][0] for i in range(len(list_row_to_dup1))])]
#we print their cardinal
print(len(df_dup))
#we print the fisrt line that ;ust have ben duplicated in the original dataframe but only the tokenization_2 column
print(df_dup[df_dup['TRIP_ID']==list_row_to_dup[0][0]]['Tokenization_2'])
#we print the lines corresponding to the fisrt line that must have been duplicated in the original dataframe but that time in the dataframe after duplication ie df_full_dup1 but only the tokenization_2 column
print(df_full_dup1[df_full_dup1['TRIP_ID']==list_row_to_dup[0][0]]['Tokenization_2'])


#we want to verify if df_full_dup1 is equal to the dataframe (permutation of the rows included) that have been formatted but without the duplication and separation (df_full)
#we need to put the rows of df_full_dup1 in the same order as the original dataframe, for that we can sort the rows of df_full_dup1 by the TRIP_ID and idem for the original dataframe
def comparison_equality_df(df_full_dup1, dataframe_origin):
    #we sort the rows of df_full_dup1 by the TRIP_ID
    df_full_dup1.sort_values(by=['TRIP_ID'], inplace=True)
    #we sort the rows of the original dataframe by the TRIP_ID
    dataframe_origin.sort_values(by=['TRIP_ID'], inplace=True)
    #we reset the index of the two dataframes
    df_full_dup1.reset_index(drop=True, inplace=True)
    dataframe_origin.reset_index(drop=True, inplace=True)
    #we compare the two dataframes
    if df_full_dup1.equals(dataframe_origin):
        return 'The two dataframes are equal'
    else:
        return 'The two dataframes are not equal'
    
a=comparison_equality_df(df_full_dup1, df_full)
