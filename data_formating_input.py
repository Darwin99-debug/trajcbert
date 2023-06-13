#importation des librairies

import pandas as pd
import json
from transformers import BertModel, BertTokenizer

#we load the json file
with open('/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json', 'r') as openfile:
    json_loaded = json.load(openfile)
data_format = pd.DataFrame(data=json_loaded)

#on vérifie que les données dans la colonne Tokenization sont bien des listes
data_format['Tokenization'].apply(lambda x: isinstance(x,list)).value_counts()
print("Verification du format des données effectuée.")
#on va mettre dans une nouvelle colonne INPUT la concaténation dans cet ordre, du jetpn CLS, du dernier élément de ligne sur la colonne Tokenization, la colonne CALL_TYPE, la colonne TAXI_ID,  et enfin la liste des éléments la ligne pour la colonne Tokenization jusqu'à l'avant avant dernier élément

#on commence par transformer les éléments de la colonne CALL_TYPE et TAXI_ID en string
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))

#on continue en ajoutant un espace devant les éléments de la colonne CALL_TYPE et TAXI_ID en prévision de la concaténation pour séparer d'un espace les éléments de la colonne Tokenization et les autres colonnes
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)


data_format['INPUT'] = data_format['Tokenization'].apply(lambda x: ['[CLS]'] + x[-1])  + data_format['CALL_TYPE'] + data_format['TAXI_ID'].apply(lambda x: str(x) + ['[SEP]']  )+ data_format['Tokenization'].apply(lambda x: ' '.join(x[:-2]))
# la target sera l'avant dernier élément de la colonne Tokenization
data_format['TARGET']=data_format['Tokenization'].apply(lambda x: x[-2])

#on supprime les colonnes inutiles
data_format.drop(['Tokenization','CALL_TYPE','TAXI_ID'],axis=1,inplace=True)

#on sauvegarde le dataframe dans un fichier json
data_format.to_json('/home/daril_kw/data/data_formated_final.json',orient='records')

