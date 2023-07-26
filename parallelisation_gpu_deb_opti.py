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

def add_spaces_for_concat(data_format, column):
    """Add spaces before and after the values of the column"""
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


def formatting_to_train(data_format, tokenizer):
    """
    Format the data to train the model : 
    ------------------------------------

    1) format the input

        a) get the full_inputs
    - we concatenate the context input and the beginning of the trajectory which is the sequence we want to give to the model 
    - at the beginning, we add the CLS token and the end of the input the SEP token

        b) get the input_ids
    - we use the tokenizer to get the ids of the tokens that will be the input_ids thatthe model will take as input
    - we pad the input to the maximum length of 512

    2) and we create the attention masks

    - the attention mask is a list of 0 and 1, 0 for the padded tokens and 1 for the other tokens

    """
    
    #we remove the useless columns
    if 'Tokenization' in data_format.columns:
        data_format.drop(['Tokenization'],axis=1,inplace=True)
    if 'CALL_TYPE' in data_format.columns:
        data_format.drop(['CALL_TYPE'],axis=1,inplace=True)
    if 'TAXI_ID' in data_format.columns:
        data_format.drop(['TAXI_ID'],axis=1,inplace=True)
    if 'DAY' in data_format.columns:
        data_format.drop(['DAY'],axis=1,inplace=True)
    if 'HOUR' in data_format.columns:
        data_format.drop(['HOUR'],axis=1,inplace=True)
    if 'WEEK' in data_format.columns:
        data_format.drop(['WEEK'],axis=1,inplace=True)
    if 'Nb_points_token' in data_format.columns:
        data_format.drop(['Nb_points_token'],axis=1,inplace=True)


    #we get the columns CONTEXT_INPUT, DEB_TRAJ and TARGET
    c_inputs=data_format.CONTEXT_INPUT.values
    traj_inputs=data_format.DEB_TRAJ.values
    targets=data_format.TARGET.values

    print("concaténation des inputs, padding etc")

    #we create the input_ids, the attention_masks and the full_inputs
    input_ids = []
    full_inputs = []
    attention_masks = []
    for i in tqdm(range(len(c_inputs))):
        #no truncation is needed because we managed it before

        #we concatenate the context input and the trajectory input adding manually the CLS token and the SEP token
        full_input = '[CLS] ' + c_inputs[i] + ' ' + traj_inputs[i] + ' [SEP]'
        full_inputs.append(full_input)

        # we use the tokenizer to get the ids of the tokens that will be the input_ids that the model will take as input
        # the format of the input_ids would be : [101] + encoded_c_input + encoded_traj_input + [102]
        #the[101] token is the CLS token and the [102] token is the SEP token
        # TODO : test adding an additional SEP token between the context input and the trajectory input so that the format of the input_ids would be : [101] + encoded_c_input + [102] + encoded_traj_input + [102]
        encoded_full_input=tokenizer.encode(full_input, add_special_tokens=False)

        #we pad the input to the maximum length of 512
        encoded_full_input=encoded_full_input + [0]*(512-len(encoded_full_input))
        #we add the input_ids to the list
        input_ids.append(encoded_full_input)

        #we create the attention mask
        att_mask = [float(i>0) for i in encoded_full_input]
        #we add the attention mask to the list
        attention_masks.append(att_mask)

    return input_ids, attention_masks, targets, full_inputs


def main():

    WORLD_S=2
    h3_config_size = 10
    nb_rows = 60

    #we load the data
    with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
        json_loaded = json.load(openfile)
    data_format = pd.DataFrame(data=json_loaded)

    #we keep only nb_rows rows
    data_format = truncation_rows(data_format, nb_rows)

    #we add the tokenization column
    data_format = add_tokenization_column(data_format, h3_config_size)

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

    #we get the tokenizer from the HuggingFace library, this one is the tokenizer of the model bert-base-cased but we could have taken the non trained tokenizer (TODO :test it with the non trained weights of the model as well)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    #we add the geographical and contextual tokens to the tokenizer so that the vocabulary of the tokenizer is adapted to our data
    tokenizer, nb_token_geo = add_geo_and_context_tokens_tokenizer(tokenizer, data_format)
    #we get the number of labels which is the number of geographical tokens + 1 (the +1 is for the [SEP] token which is for the end of the sequence and the prediction)
    nb_labels = nb_token_geo + 1

    #we get the model from the HuggingFace library, this one is the model bert-base-cased but we could have taken the non trained model (if we want to train it from scratch)
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=nb_labels)
    #we add the geographical and contextual tokens to the model so that the size of the model`s embedding is adapted to our data
    model.resize_token_embeddings(len(tokenizer))

    #save the model, the tokenizer and the data in different files
    model.save_pretrained(f"/home/daril_kw/data/model_before_training_opti_{nb_rows}")
    data_format.to_json(f"/home/daril_kw/data/data_with_time_info_ok_opti2_{nb_rows}.json")
    tokenizer.save_pretrained(f"/home/daril_kw/data/tokenizer_final_opti_{nb_rows}")

    #we get the DEB_TRAJ and TARGET columns well formatted but without the special tokens [CLS] and [SEP]
    data_format = get_deb_traj_and_target(data_format)

    #we get the input_ids, the attention_masks, the targets and the full_inputs
    input_ids, attention_masks, targets, full_inputs = formatting_to_train(data_format, tokenizer)
    
    #save the lists full_inputs, inputs_ids, attention_masks and the targets in different files
    with open(f"/home/daril_kw/data/input_ids_{nb_rows}_opti.pkl", 'wb') as fp:
        pickle.dump(input_ids, fp)
    with open(f"/home/daril_kw/data/attention_masks_{nb_rows}_opti.pkl", 'wb') as fp:
        pickle.dump(attention_masks, fp)
    with open(f"/home/daril_kw/data/targets_{nb_rows}_opti.pkl", 'wb') as fp:
        pickle.dump(targets, fp)
    with open(f"/home/daril_kw/data/full_inputs_{nb_rows}_opti.pkl", 'wb') as fp:
        pickle.dump(full_inputs, fp)


if __name__ == "__main__":
    main()