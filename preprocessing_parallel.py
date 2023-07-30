import json
import datetime
import os
import pickle
import h3
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from concurrent.futures import ThreadPoolExecutor

def truncation_rows(df, nb_rows):
    return df[:nb_rows]


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


# Rest of the code remains the same until the `main()` function

def process_row(row, tokenizer, h3_config_size):
    # Add tokenization column
    row['Tokenization_2'] = [h3.geo_to_h3(coord[1], coord[0], h3_config_size) for coord in row['POLYLINE']]

    # Add time information
    row['DATE'] = datetime.datetime.fromtimestamp(row['TIMESTAMP']).strftime('%Y-%m-%d %H:%M:%S')
    row['DAY'] = str(datetime.datetime.strptime(row['DATE'].split(' ')[0], '%Y-%m-%d').isocalendar()[2])
    row['HOUR'] = row['DATE'].split(' ')[1].split(':')[0]
    row['WEEK'] = str(datetime.datetime.strptime(row['DATE'].split(' ')[0], '%Y-%m-%d').isocalendar()[1])

    # Remove unnecessary columns
    row = row.drop(['MISSING_DATA', 'DATE', 'ORIGIN_CALL', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP'])

    # Convert columns to string types
    row['TAXI_ID'] = str(row['TAXI_ID'])
    row['CALL_TYPE'] = str(row['CALL_TYPE'])

    # Transform CALL_TYPE to a number
    row['CALL_TYPE'] = 0 if row['CALL_TYPE'] == 'A' else (1 if row['CALL_TYPE'] == 'B' else 2)

    return row

def main():
    WORLD_S = 2
    h3_config_size = 10
    nb_rows = 60

    # Load the data
    with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
        json_loaded = json.load(openfile)
    data_format = pd.DataFrame(data=json_loaded)

    # Keep only nb_rows rows
    # data_format = truncation_rows(data_format, nb_rows)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Process each row concurrently
        processed_rows = list(tqdm(executor.map(lambda row: process_row(row, tokenizer, h3_config_size), data_format.iterrows(), chunksize=10), total=len(data_format)))
        # the chunksize parameter is used to specify how many rows to process at a time 
        # it depends on the number of cores of your machine

    # Create a new DataFrame from the processed rows
    data_format = pd.DataFrame(processed_rows)

    # Get the tokenizer from the HuggingFace library
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Add the geographical and contextual tokens to the tokenizer
    tokenizer, nb_token_geo = add_geo_and_context_tokens_tokenizer(tokenizer, data_format)

    # Get the number of labels
    nb_labels = nb_token_geo + 1

    # Get the model from the HuggingFace library
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=nb_labels)

    # Add the geographical and contextual tokens to the model
    model.resize_token_embeddings(len(tokenizer))

    # Save the model, the tokenizer, and the data in different files
    model.save_pretrained(f"/home/daril_kw/data/model_before_training_opti_full")
    data_format.to_json(f"/home/daril_kw/data/data_with_time_info_ok_opti2_full.json")
    tokenizer.save_pretrained(f"/home/daril_kw/data/tokenizer_final_opti_full")

    # Get the DEB_TRAJ and TARGET columns well formatted but without the special tokens [CLS] and [SEP]
    data_format = get_deb_traj_and_target(data_format)

    # Get the input_ids, the attention_masks, the targets, and the full_inputs
    input_ids, attention_masks, targets, full_inputs = formatting_to_train(data_format, tokenizer)
    
    # Save the lists full_inputs, inputs_ids, attention_masks, and the targets in different files
    with open(f"/home/daril_kw/data/input_ids_full_opti.pkl", 'wb') as fp:
        pickle.dump(input_ids, fp)
    with open(f"/home/daril_kw/data/attention_masks_full_opti.pkl", 'wb') as fp:
        pickle.dump(attention_masks, fp)
    with open(f"/home/daril_kw/data/targets_full_opti.pkl", 'wb') as fp:
        pickle.dump(targets, fp)
    with open(f"/home/daril_kw/data/full_inputs_full_opti.pkl", 'wb') as fp:
        pickle.dump(full_inputs, fp)

if __name__ == "__main__":
    main()