import pandas as pd
import json
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

print("We open the dataset in the json file.")


#we read the data
with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:

    # Reading from json file
    json_loaded = json.load(openfile)

print("We put the data in a dataset.")


#we put them in a dataframe
data_clean = pd.DataFrame(data=json_loaded)

#we want to put in a list all the different token from the lists in the Tokenization column

print("We create the list of tokens.")
list_token = []
for i in range(len(data_clean)):
    list_token += data_clean['Tokenization'][i]

#we keep every token only once
list_token = list(set(list_token))

#we save in in a txt file
print("We save the list of tokens in a txt file.")
with open('/home/daril_kw/data/list_geographic_token.txt', 'w') as f:
    for item in list_token:
        f.write("%s\n" % item)

#len(list_token) = 121089


print("We load the model and the tokenizer.")
#we load the model and the tokenizer
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/trajcbert/trajcbert/new_tokenizer')

model = BertModel.from_pretrained("bert-base-cased")


print("We create the list of new tokens.")

timestamp_tokens =  []
for i in range(len(data_clean)):
    timestamp_tokens.append(str(data_clean['TIMESTAMP'][i]))

timestamp_tokens = list(set(timestamp_tokens))

print(f"Tokenizer size bfr adding the timestamp tokens (context): {len(tokenizer)}")

print("We add the tokens for the timestamp on the tokenizer.")


for token in tqdm(timestamp_tokens):
    tokenizer.add_tokens(token)

print(f"Tokenizer size after adding the timestamp tokens: {len(tokenizer)}")



model.resize_token_embeddings(len(tokenizer))

print("we save the new tokenizer")
tokenizer.save_pretrained('/home/daril_kw/data/tokenizer_full')
