import pandas as pd
import json
import torch
from transformers import BertModel, BertTokenizer

#we read the data
with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
 
    # Reading from json file
    json_loaded = json.load(openfile)
 
# print(json_loaded)
print(type(json_loaded))

#we put them in a dataframe
data_clean = pd.DataFrame(data=json_loaded)

#we load the model and the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased") 

model = BertModel.from_pretrained("bert-base-cased") 

  

#we find the token we want to add to the tokenizer
#for that, we put in a list all the tokens of the lists ins the column 'Tokenization' of the dataframe
list_token = []
for i in range(len(data_clean)):
    for j in range(len(data_clean['Tokenization'][i])):
        list_token.append(data_clean['Tokenization'][i][j])

#we remove the duplicates
list_token = list(set(list_token))
    
print(f"Tokenizer size bfr adding the new tokens : {len(tokenizer)}")#  28996
#we add the new tokens to the tokenizer

tokenizer.add_tokens(list_token)

print(f"Tokenizer size after adding the new tokens : {len(tokenizer)}")# 121089 (len(list_token)) + 28996 = 150085

  
#we adapt the size of the embedding
model.resize_token_embeddings(len(tokenizer))  

# Randomly generated matrix 
#for i in range(len(list_token)):
#    with torch.no_grad():
#        model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size]) 


  
#print(" New embedding vector at the end of the embedding matrix")
#print(model.embeddings.word_embeddings.weight[-1, :])



#we save the new tokenizer
tokenizer.save_pretrained('/home/daril_kw/data/02.06.23/tokenizer_clean')   

#we can add the token of the contextual information we want ie the following columns of the dataframe : 'Call_type', 'Taxi_id', 'Timestamp'
#but we need them all of them as strings and not as integers so we need to convert them except for the column 'Call_type' which is already a string

contextual_info_token = []
for i in range(len(data_clean)):
    contextual_info_token.append(data_clean['CALL_TYPE'][i])
    contextual_info_token.append(str(data_clean['TAXI_ID'][i]))
    contextual_info_token.append(str(data_clean['TIMESTAMP'][i]))   

#we remove the duplicates
contextual_info_token = list(set(contextual_info_token))
    

#we add the new tokens to the tokenizer 
print(f"Tokenizer size bfr adding the new tokens (context): {len(tokenizer)}")# 150085
tokenizer.add_tokens(contextual_info_token)

print(f"Tokenizer size after adding the new tokens: {len(tokenizer)}")# 

  

model.resize_token_embeddings(len(tokenizer))  

# The new vector is added at the end of the embedding matrix 

  
#print(" Old embedding vector at the end of the embedding matrix")
#print(model.embeddings.word_embeddings.weight[-1, :]) 

# Randomly generated matrix 

#for i in range(len(contextual_info_token)):
#    with torch.no_grad():
#        model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size]) 

  

