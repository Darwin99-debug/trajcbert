#importation des librairies

import pandas as pd
import json
from transformers import BertModel, BertTokenizer

#ouverture du fichier json

with open('/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json', 'r') as openfile:
    json_loaded = json.load(openfile)

#on met les données dans un dataframe
print("Chargement effectué.")

data_clean = pd.DataFrame(data=json_loaded)

# on fait une copie du dataframe
data_format=data_clean.copy()

#on vérifie que les données dans la colonne Tokenization sont bien des listes
data_format['Tokenization'].apply(lambda x: isinstance(x,list)).value_counts()
print("Verification du format des données effectuée.")
#on va mettre dans une nouvelle colonne INPUT la concaténation dans cet ordre, du dernier élément de ligne sur la colonne Tokenization, la colonne CALL_TYPE, la colonne TAXI_ID,  et enfin la liste des éléments la ligne pour la colonne Tokenization jusqu'à l'avant avant dernier élément
#on sépare les éléments de contxete (derneir élément de la colonne Tokenization, et les autres colonnes) du débu de la tokenization par <SEP>

#on commence par transformer les éléments de la colonne CALL_TYPE et TAXI_ID en string
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))

#on continue en ajoutant un espace devant les éléments de la colonne CALL_TYPE et TAXI_ID en prévision de la concaténation pour séparer d'un espace les éléments de la colonne Tokenization et les autres colonnes
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)

#on veut ensuite gérer la colonne tokenization pour la concaténation
# en effet on veut concaténer les éléments de la liste de la colonne Tokenization, mais pas le dernier élément, qui sera la entré en contxete au début, et pas l'avant dernier élément qui sera la cible


data_format['INPUT'] = data_format['Tokenization'].apply(lambda x: x[-1])  + data_format['CALL_TYPE'] + data_format['TAXI_ID'].apply(lambda x: str(x)) + '<SEP>' + data_format['Tokenization'].apply(lambda x: ' '.join(x[:-2]))
# la target sera l'avant dernier élément de la colonne Tokenization
data_format['TARGET']=data_format['Tokenization'].apply(lambda x: x[-2])

#on supprime les colonnes inutiles
data_format.drop(['Tokenization','CALL_TYPE','TAXI_ID'],axis=1,inplace=True)

#on sauvegarde le dataframe dans un fichier json
data_format.to_json('/home/daril_kw/data/data_formated.json',orient='records')





