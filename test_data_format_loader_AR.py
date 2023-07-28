#libraries
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tqdm import tqdm



#directories :
#-------------

#loading
data_test_dir='/home/daril_kw/data/data_test_gene_AR_to_format.pkl'
tokenizer_dir = '/home/daril_kw/data/tokenizer_final'
targets_dict_dir='/home/daril_kw/data/AR/targets_dict_v_small_AR.pt'
targets_input_dir= '/home/daril_kw/data/AR/targets_input_v_small_AR.pt'


#parameters :
#------------
batch_size = 32

#------------------------------------------------------------------------------------

def get_traj(dataframe):
    len_context_info = len(dataframe['CONTEXT_INPUT'][0].split(' '))

    """the TRAJ column will be the tokenization column truncated"""
    dataframe['TRAJ']=dataframe['Tokenization_2']

    # we manage the length of the CONTEXT_INPUT column so that after the concatenation, it does not exceed 512 tokens
    # the -2 corresponds to the two special tokens [CLS] and [SEP]
    # for exemple here, if the trajectory input is too long, we keep the 512-6-2=504 last tokens
    dataframe['TRAJ']=dataframe['TRAJ'].apply(lambda x: x[-(512-len_context_info-2):] if len(x)>512-len_context_info-2 else x)    

    #then we keep the column in form of a string with spaces between the tokens (the space replaces the comma)
    dataframe['TRAJ']=dataframe['TRAJ'].apply(lambda x: ' '.join(x))

    return dataframe


def get_whole_inputs(dataframe):
    dataframe_original = dataframe
    dataframe = dataframe_original.copy()
    """We concatenate the CONTEXT_INPUT and the TRAJ columns and we add a space between them + we add the special tokens [CLS] and [SEP]"""
    for i in tqdm(range(len(dataframe))):
        #same with using iloc :
        dataframe['WHOLE_INPUT'].iloc[i] = '[CLS] ' + dataframe['CONTEXT_INPUT'].iloc[i] + ' ' + dataframe['TRAJ'].iloc[i] + ' [SEP]'


    return dataframe


def prepare_data(input, tokenizer, targets_input):
    input_sequences = []
    
    
    for idx, input_seq in enumerate(input):
    
        # Encode the text.
        encoded_sequence = tokenizer.encode(input_seq, add_special_tokens=False, padding=False)
        input_sequences.append(encoded_sequence)

    # Convert to tensors.
    inputs = torch.tensor(input_sequences)

    return inputs




def main():

    #loading the data in a list
    with open(data_test_dir, 'rb') as f:
        data_test = pickle.load(f)

    #loading the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

    #loading the targets_dict and the targets_input
    targets_dict = torch.load(targets_dict_dir)
    targets_input = torch.load(targets_input_dir)
    #the targets_input is a dictionnary with the token as key and the number as value
    #the use is the following : targets_input[token] = number where token is a string and number is an int
    
    #we add the columns that we need
    data_test = get_traj(data_test)
    data_test = get_whole_inputs(data_test)

    inputs = []
    for i in tqdm(range(len(data_test))):
        inputs.append(data_test['WHOLE_INPUT'][i].tolist())

    #we remove the columns that we don't need anymore
    if 'Tokenization' in data_test.columns:
        data_test.drop(['Tokenization'],axis=1,inplace=True)
    if 'CALL_TYPE' in data_test.columns:
        data_test.drop(['CALL_TYPE'],axis=1,inplace=True)
    if 'TAXI_ID' in data_test.columns:
        data_test.drop(['TAXI_ID'],axis=1,inplace=True)
    if 'DAY' in data_test.columns:
        data_test.drop(['DAY'],axis=1,inplace=True)
    if 'HOUR' in data_test.columns:
        data_test.drop(['HOUR'],axis=1,inplace=True)
    if 'WEEK' in data_test.columns:
        data_test.drop(['WEEK'],axis=1,inplace=True)
    if 'Nb_points_token' in data_test.columns:
        data_test.drop(['Nb_points_token'],axis=1,inplace=True)

    
    #we prepare the data for the model
    inputs_ids = prepare_data(inputs, tokenizer, targets_input)

    #we create the dataloader
    prediction_data = TensorDataset(inputs_ids)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data,sampler=prediction_sampler, batch_size=batch_size)

    return prediction_dataloader, data_test, targets_dict, targets_input
    


if __name__ == "__main__":
    main()