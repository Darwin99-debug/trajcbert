import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import json



def add_spaces_for_concat(data_format, column):
    """Add spaces before and after the values of the column"""
    data_format[column]=data_format[column].apply(lambda x: ' '+x)
    return data_format



def rows_attribution_cat(dataframe, nb_categories):
    """Distribute the rows of the dataframe among categories"""
    nb_rows_per_cat = len(dataframe) // nb_categories
    nb_rows_dict = {f'nb_rows_category{i}': nb_rows_per_cat for i in range(nb_categories)}
    remainder = len(dataframe) % nb_categories
    
    # Distribute the remaining rows randomly among categories
    for i in random.sample(range(nb_categories), remainder):
        nb_rows_dict[f'nb_rows_category{i}'] += 1
    
    return nb_rows_dict

def create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict):
    """Create a dictionary of dataframes, each dataframe corresponding to a category"""
    df_dict = {f'dataframe_category{i}': dataframe.iloc[list_index_dict[f'list_index_category{i}']].copy() for i in range(nb_categories)}
    for i in range(nb_categories):
        df_dict[f'dataframe_category{i}'] = df_dict[f'dataframe_category{i}'].reindex(columns=dataframe.columns)
    return df_dict

def create_target_deb_traj(nb_categories, df_dict):
    """Create a dictionary of lists of targets and a dictionary of lists of deb_traj"""
    target_dict = {f'list_target_category{i}': np.empty(len(df_dict[f'dataframe_category{i}']), dtype=object) for i in range(nb_categories)}
    list_deb_traj_dict = {f'list_deb_traj_category{i}': [[] for _ in range(len(df_dict[f'dataframe_category{i}']))] for i in range(nb_categories)}
    return target_dict, list_deb_traj_dict

def fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict):
    """Fill the target and deb_traj lists"""

    #Manage the first categories
    for i in range(nb_categories-2):
        df = df_dict[f'dataframe_category{i}']
        for j in range(len(df)):
            tokenization_2 = df.iloc[j]['Tokenization_2']
            start_idx = int(list_threshold[i] * len(tokenization_2))
            end_idx = int(list_threshold[i+1] * len(tokenization_2))
            tokenization_2 = tokenization_2[start_idx:end_idx]
            
            # in case the trajectory is too short to be split, we take the last token of the trajectory
            if len(tokenization_2) != 0:
                index = random.randint(0, len(tokenization_2)-1)
            else:
                index = -1
            
            token = tokenization_2[index]
            target_dict[f'list_target_category{i}'][j] = token
            list_deb_traj_dict[f'list_deb_traj_category{i}'][j].extend(df.iloc[j]['Tokenization_2'][:start_idx])
            list_deb_traj_dict[f'list_deb_traj_category{i}'][j].extend(tokenization_2[:index])

    #Manage the last two categories
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

def manage_duplication(dataframe, liste_to_duplicate):
    # Convert liste_to_duplicate elements to tuples and create a set
    liste_to_duplicate_trip_id = [item[0] for item in liste_to_duplicate]

    # Create a dataframe to store duplicated rows
    duplicated_rows = pd.DataFrame()

    # Duplicate rows for each unique TRIP_ID value. we use the 2nd argument of each sublist of liste_to_duplicate to know how many times we duplicate the row
    for trip_id in liste_to_duplicate_trip_id:
        #we find the row that have the same TRIP_ID
        df = dataframe[dataframe['TRIP_ID'] == trip_id]
        #we add the row to the dataframe of duplicated rows
        for i in range(liste_to_duplicate[liste_to_duplicate_trip_id.index(trip_id)][1]-1):
            duplicated_rows = pd.concat([duplicated_rows, df], ignore_index=True)
        
    
    #we add the duplicated rows to the dataframe
    return pd.concat([dataframe, duplicated_rows], ignore_index=True)



def attribution_deb_traj_and_target(dataframe, nb_categories=5, liste_to_duplicate=[], decal_gauche=False, decal_droite=False, uniforme=True):

    """Prepare the training data without duplicates
    liste_to_duplicate is a list of TRIP_ID that we want to duplicate 
    we create the threshold for each category knowing that they go from 0.3 to 1 (the last token is excluded)
    tow categories are reserved for the last token (the destination) and the [SEP] token so we don't take them into account
    for example, if ze have 5 categories, the uniform threshold would be (1-0.3)/(5-2) = 0.23333333333333334
    that means that the first category will concern length of trajectory from 0.3 to 0.5333333333333333, the second from 0.5333333333333333 to 0.7666666666666666 and the third from 0.7666666666666666 to 1
    we create a list of threshold"""

    dataframe_original = dataframe
    dataframe = dataframe_original.copy()

    # Create the threshold for each category
    list_threshold = [0.3 + i * ((1 - 0.3) / (nb_categories - 2)) for i in range(nb_categories - 1)]

    # Remove the useless rows and rows with trajectory length < 3
    dataframe.loc[:, 'Tokenization_2'] = dataframe['Tokenization_2'].apply(lambda x: x if type(x) == list else [])
    dataframe.loc[:, "LEN_TRAJ"] = dataframe['Tokenization_2'].apply(lambda x: len(x))
    dataframe = dataframe[dataframe['LEN_TRAJ'] >= 3]

    

    # Create a seed to be able to reproduce the results
    random.seed(2023)
    # Create a dictionary of the number of rows per category
    nb_rows_dict = rows_attribution_cat(dataframe, nb_categories)
    # Create a dictionary of lists of indexes of rows per category
    list_index_dict = {f'list_index_category{i}': np.array(random.sample(range(len(dataframe)), nb_rows_dict[f'nb_rows_category{i}'])) for i in range(nb_categories)}

    # Create a dictionary of dataframes, each dataframe corresponding to a category
    df_dict = create_df_cat(dataframe, nb_categories, nb_rows_dict, list_index_dict)

    # Create a dictionary of lists of targets and a dictionary of lists of deb_traj
    target_dict, list_deb_traj_dict = create_target_deb_traj(nb_categories, df_dict)
    target_dict, list_deb_traj_dict = fill_target_deb_traj(df_dict, nb_categories, list_threshold, target_dict, list_deb_traj_dict)

    # Add the target and deb_traj columns to each dataframe
    for i in range(nb_categories):
        df_dict[f'dataframe_category{i}']['TARGET'] = target_dict[f'list_target_category{i}']
        df_dict[f'dataframe_category{i}']['DEB_TRAJ'] = list_deb_traj_dict[f'list_deb_traj_category{i}']
    
    # Concatenate the dataframes and return the result
    return pd.concat([df_dict[f'dataframe_category{i}'] for i in range(nb_categories)], ignore_index=True)


def manage_separation(dataframe, list_index_to_separate):
    """
    This function separates the rows of the dataframe according to the list of rows to separate
    """
    dataframe_separated = dataframe.copy()
    dataframe_separated.reset_index(drop=True, inplace=True)

    # Create a dictionary to map TRIP_ID to its index in list_index_to_separate
    trip_id_to_index = {trip_id: i for i, (trip_id, _) in enumerate(list_index_to_separate)}

    # Sort list_index based on the order of TRIP_ID in list_index_to_separate
    list_index = sorted(dataframe_separated.index, key=lambda idx: trip_id_to_index.get(dataframe_separated.loc[idx, 'TRIP_ID'], float('inf')))

    modified_rows = []

    for i in range(len(list_index_to_separate)):
        row = dataframe_separated.loc[list_index[i]].copy()
        dataframe_separated = dataframe_separated.drop(list_index[i]) 

        # Separate the trajectory into nb_traj sub-trajectories
        list_traj = []
        tokenization_2 = row['Tokenization_2']
        len_traj = len(tokenization_2)
        nb_traj = list_index_to_separate[i][1]
        len_each_traj = len_traj // nb_traj
        # Create nb_traj sub-trajectories of equal length if possible
        for j in range(nb_traj):
            traj = tokenization_2[j * len_each_traj:(j + 1) * len_each_traj]
            list_traj.append(traj)
        rest = len_traj % nb_traj
        # Add the remaining points to the last sub-trajectory (if any, which is the case if len_traj is not a multiple of nb_traj)
        if rest != 0:
            for point in tokenization_2[-rest:]:
                list_traj[-1].append(point)

        # Create a new row for each sub-trajectory
        for j in range(nb_traj):
            new_row = row.copy()
            new_row['Tokenization_2'] = list_traj[j]
            modified_rows.append(new_row)

    # Remove the rows that we separated
    for i in range(len(list_index_to_separate)):
        idx_to_remove = [idx for idx in dataframe_separated.index if dataframe_separated.loc[idx, 'TRIP_ID'] == list_index_to_separate[i][0]]
        dataframe_separated = dataframe_separated.drop(idx_to_remove)  # Use .loc[] here

    # Add the new rows
    for row in modified_rows:
        dataframe_separated = pd.concat([dataframe_separated, row.to_frame().T])

    dataframe_separated.reset_index(drop=True, inplace=True)

    return dataframe_separated

def row_selection(dataframe, nb_to_select):
    """
    This function selects the rows that we will separate or duplicate without makimg any distinction
    between the two treatments for now
    """
    #we create a seed to be able to reproduce the results
    random.seed(2023)

    #we select the rows we are going to separate or duplicate according to their length : we select the rows with the longest trajectories
    #for that, we sort the dataframe by the length of the trajectory
    #the thing is, the two part of a trajectory can be longer than the following longest trajectory
    #that means we can have the longest trajectory of length 512 for example, wich means the two resulting trajectories will be of length 256
    #but the 2nd longest trajectory is of length 255, so the two resulting trajectories will be longer than the 2nd longest trajectory
    # that is why we need to sort at each iteration
    #we create a dataframe that zill be the fisrt dataframe but sorted by the length of the trajectory and we keep the matching before sorting to ne able to find the rows in the original dataframe
    dataframe.loc[:, "LEN_TRAJ"] = dataframe['Tokenization_2'].apply(lambda x: len(x))
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
    
    return list_row_to_select


def attribution_duplicate_or_separate(list_row_to_select, nb_to_duplicate, nb_to_separate):
    """
    This function attributes the rows that we will separate and the rows that we will duplicate
    knowing the number of rows that we will separate and the number of rows that we will duplicate
    and the list of rows that we will separate or duplicate
    """
    #we create a list of index of the rows that we will separate
    list_index_to_separate = []
    #we create a list of index of the rows that we will duplicate
    list_index_to_duplicate = []
    #we choose randomly the rows that we will separate and the rows that we will duplicate
    list_index_to_separate = random.sample(list_row_to_select, nb_to_separate)
    #we take the rows that we did not select for separation
    list_index_to_duplicate = [i for i in list_row_to_select if i not in list_index_to_separate]


def prepare_train(dataframe, duplication_rate=0, separation_rate=50):
    """
    This function prepares the train dataset like the prepare_train_wo_duplicate function but with the possibility to duplicate the rows.
    The separation rate is the proportion of rows that will separated into two different trajectories. 
    The duplication rate is the proportion of rows that will be duplicated, ie that will occur in two different trajectories with different targets.
    """

    #we copy to avoid caveat
    dataframe_original = dataframe
    dataframe = dataframe_original.copy()

    #we calculate the number of rows that we will separate and the number of rows that we will duplicate
    nb_to_separate = int(len(dataframe)*separation_rate/100)
    nb_to_duplicate = int(len(dataframe)*duplication_rate/100)
    nb_to_select=nb_to_separate+nb_to_duplicate

    #we select the rows we are going to separate or duplicate with the number of times we will duplicate them or separate them
    list_row_to_select = row_selection(dataframe, nb_to_select)

    #we attribute the rows that we will separate and the rows that we will duplicate among the rows that we selected
    list_index_to_duplicate, list_index_to_separate = attribution_duplicate_or_separate(list_row_to_select, nb_to_duplicate, nb_to_separate)

    #we separate the rows that were chosen to be separated
    dataframe_separated=manage_separation(dataframe, list_index_to_separate)

    #we duplicate the rows that were chosen to be duplicated
    dataframe_sep_and_dup = manage_duplication(dataframe_separated, list_index_to_duplicate)

    #we attribute the target and the deb_traj to the rows
    df_full = attribution_deb_traj_and_target(dataframe_sep_and_dup)

    return df_full, dataframe_separated, list_index_to_separate, list_index_to_duplicate


def verif_separation(dataframe, list_row_to_sep):
    """
    we verify that the rows that we separated are well separated
    the idea is to verify that the number of rows that we separated is equal to the number of sub-trajectories 
    that we created from the original trajectory
    """
    for j in range(len(list_row_to_sep)):
        if len(dataframe[dataframe['TRIP_ID']==list_row_to_sep[j][0]])!=list_row_to_sep[j][1]:
            raise ValueError('The rows are not well separated')
    return 'The rows are well separated'




def verif_concatenation(df_full, df_sep):
    """
part of the verification is to see whether the concatenation of the subtrajectories is equal to the original trajectory
for that, we count whether the number of points of the concatenation is equal to the number of points of the original trajectory (column Tokenization_2 of df_full)
    """
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
        #we concatenate the tokenization_2 of the rows of df_sep for a trip_id and compare it to the tokenization_2 of the row of df_full for the same trip_id
        #we cannot use the join function because the tokenization_2 are lists
        concatenation = []
        for j in range(len(df)):
            concatenation.extend(df.iloc[j]['Tokenization_2'])
        if concatenation != df_full['Tokenization_2'][i]:
            print('concatenation : ', concatenation)
            print('______________________________________________')
            print('df_full : ', df_full['Tokenization_2'][i])
            raise ValueError('The concatenation of the trajectories is not equal to the original trajectory')
    return 'The nb of points resuting from the concatenation of the trajectories is equal to the nb of points in the original trajectory'



def verif_length(dataframe, list_row_to_sep, list_row_to_dup):
    """
we verify that the dataframe obtained with prepare_train as the good length
what we call good length is :
   its original length
 + the number of rows that we duplicated * (the number of duplication) 
 - the number of rows that we duplicated 
 + the number of rows that we separated * the number of sub-trajectories that we created from the original trajectory
 - the number of rows that we separated
    """
    if len(dataframe) != len(data_train) + sum([list_row_to_sep[i][1] for i in range(len(list_row_to_sep))]) - len(list_row_to_sep) + sum([list_row_to_dup[i][1] for i in range(len(list_row_to_dup))]) - len(list_row_to_dup):
        raise ValueError('The dataframe does not have the good length')
    return 'The dataframe has the good length'

 
if __name__ == '__main__':

    #load the tokenizer from /home/daril_kw/data/tokenizer_final
    tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_final')
    #load the dataset from home/daril_kw/data/data_with_time_info_ok.json
    with open('/home/daril_kw/data/data_with_time_info_ok.json', 'r') as openfile:
        json_loaded = json.load(openfile)
    

    data_format = pd.DataFrame(data=json_loaded)
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

    #we call the function
    df_full = attribution_deb_traj_and_target(data_train)
    #we call the function
    df_full2, df_sep, list_row_to_sep, unused = prepare_train(data_train, duplication_rate=0, separation_rate=50)
    verif_separation(df_sep, list_row_to_sep)
    verif_concatenation(df_full, df_sep)   
    #on pase à la duplication
    df_full_dup, df_sep_dup, list_row_to_sep_dup, list_row_to_dup = prepare_train(data_train, duplication_rate=30, separation_rate=50)
    df_full_dup1, df_sep_dup1, list_row_to_sep_dup1, list_row_to_dup1 = prepare_train(data_train, duplication_rate=50, separation_rate=0)
    verif_length(df_full_dup, list_row_to_sep_dup, list_row_to_dup)
    verif_length(df_full_dup1, list_row_to_sep_dup1, list_row_to_dup1)