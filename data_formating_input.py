#importation des librairies

import pandas as pd
import json
from transformers import BertModel, BertTokenizer

#we load the json file
with open('/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json', 'r') as openfile:
    json_loaded = json.load(openfile)
data_format = pd.DataFrame(data=json_loaded)





#we want to extract three informations from the timestamp column : the day of the week, the hour and the week of the year
#we create a new column for each of them
#the timestamp is Unix timestamp ie the number of seconds since 1st January 1970
#we use the datetime library to convert the timestamp into a date
import datetime
data_format['DATE'] = data_format['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
#then we get in new columns the wanted information in a string format
data_format['DAY'] = data_format['DATE'].apply(lambda x: x.split(' ')[0])
data_format['HOUR'] = data_format['DATE'].apply(lambda x: x.split(' ')[1])
data_format['WEEK'] = data_format['DATE'].apply(lambda x: datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1])


"""data_format['DAY_OF_WEEK'] = data_format['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%A'))
data_format['HOUR'] = data_format['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%H'))
data_format['WEEK_OF_YEAR'] = data_format['DATE'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%W'))"""









#on vérifie que les données dans la colonne Tokenization sont bien des listes
data_format['Tokenization'].apply(lambda x: isinstance(x,list)).value_counts()
print("Verification du format des données effectuée.")
#on va mettre dans une nouvelle colonne INPUT la concaténation dans cet ordre, du jetpn CLS, du dernier élément de ligne sur la colonne Tokenization, la colonne CALL_TYPE, la colonne TAXI_ID,  et enfin la liste des éléments la ligne pour la colonne Tokenization jusqu'à l'avant avant dernier élément

#on commence par transformer les éléments de la colonne TAXI_ID en string
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))

#on continue en ajoutant un espace devant les éléments de la colonne HOUR, WEEK, Tokenization, CALL_TYPE, TAXI_ID en prévision de la concaténation pour séparer d'un espace les éléments de la colonne Tokenization et les autres colonnes
data_format['HOUR']=data_format['HOUR'].apply(lambda x: ' '+x)
data_format['WEEK']=data_format['WEEK'].apply(lambda x: ' '+x)
data_format['Tokenization']=data_format['Tokenization'].apply(lambda x: ' '+' '.join(x))
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)

# la colonne INPUT sera la concaténation de la colonne Tokenization jusqu'a l'avant-dernier token exclu, du jour de la semaine, de l'heure et de la semaien de l'année pui de la colonne CALL_TYPE, de la colonne TAXI_ID et du dernier token de la colonne Tokenization
data_format['INPUT'] = data_format['Tokenization'].apply(lambda x: ' '.join(x[:-2]))+ data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID'] +data_format['Tokenization'].apply(lambda x: x[-1])
# la target sera l'avant dernier élément de la colonne Tokenization
data_format['TARGET']=data_format['Tokenization'].apply(lambda x: x[-2])

#on supprime les colonnes inutiles
data_format.drop(['Tokenization','CALL_TYPE','TAXI_ID'],axis=1,inplace=True)

#on sauvegarde le dataframe dans un fichier json
data_format.to_json('/home/daril_kw/data/data_formated_final.json',orient='records')

