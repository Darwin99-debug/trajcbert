import pandas as pd
import json
import datetime


with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
    json_loaded = json.load(openfile)


data_clean = pd.DataFrame(data=json_loaded)

#on récupère la date à partir du timestamp
data_clean['DATE'] = data_clean['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

#a partir de cela on récupère le jour de la semaine sous forme d'un nombre entre 1 et 7 pour lundi à dimanche
data_clean['DAY'] = data_clean['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[2]))
#ensuite, on récupère l'heure sous forme d'un nombre entre 0 et 23
data_clean['HOUR'] = data_clean['DATE'].apply(lambda x: str(x.split(' ')[1].split(':')[0]))
#enfin on recupère le numéro de la semaine dans l'année
data_clean['WEEK'] = data_clean['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1]))


data_clean.drop(['MISSING_DATA','DATE','ORIGIN_CALL','TRIP_ID', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP' ],axis=1,inplace=True)

#on transforme la colonne CALL_TYPE en nombre
data_clean['CALL_TYPE'] = data_clean['CALL_TYPE'].apply(lambda x: 1 if x=='A' else 2 if x=='B' else 3)


from transformers import BertTokenizer, BertForSequenceClassification, BertModel
#on calcule le nombre de tokens différents dans les liste de tokens de la colonne Tokenization
list_token = []
for i in range(len(data_clean)):
    for j in range(len(data_clean['Tokenization'][i])):
        list_token.append(data_clean['Tokenization'][i][j])
list_token
nb_labels=(len(list_token))

#on load le tokenizer et le model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased") 
model = BertModel.from_pretrained("bert-base-cased") 
model_classif = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=nb_labels)

#on ajoute au tokenizer les tokens de la liste de tokens
tokenizer.add_tokens(list_token)
#on redimensionne les modèles pour qu'il puisse prendre en compte les nouveaux tokens
model.resize_token_embeddings(len(tokenizer))
model_classif.resize_token_embeddings(len(tokenizer))

#on fait de meme avec les données de contexte 
contextual_info_token = []
for i in range(len(data_clean)):
    contextual_info_token.append(data_clean['CALL_TYPE'][i])
    contextual_info_token.append(str(data_clean['TAXI_ID'][i]))
    contextual_info_token.append(data_clean['DAY'][i])
    contextual_info_token.append(data_clean['HOUR'][i])
    contextual_info_token.append(data_clean['WEEK'][i])
      

#we remove the duplicates
contextual_info_token = list(set(contextual_info_token))
tokenizer.add_tokens(contextual_info_token)
model.resize_token_embeddings(len(tokenizer))
model_classif.resize_token_embeddings(len(tokenizer))

#on sauvegarde le tokenizer
tokenizer.save_pretrained('/home/daril_kw/data/tokenizer_full')
#on sauvegarde le model
model.save_pretrained('/home/daril_kw/data/model_full')
model_classif.save_pretrained('/home/daril_kw/data/model_classif_full')
