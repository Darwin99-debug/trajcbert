import pandas as pd
import json
from transformers import BertModel, BertTokenizer


with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:
    json_loaded = json.load(openfile)

data_clean = pd.DataFrame(data=json_loaded)

new_data=data_clean.drop(['TRIP_ID','ORIGIN_CALL','ORIGIN_STAND','DAY_TYPE','MISSING_DATA','Nb_points_token'],axis=1)
print(new_data.info(), "on va sauvegarder les nveaux dataframe en json mtn")

dict_to_save=new_data.to_dict()
with open("dataset_wo_useless_columns.json", "w") as outfile:
    json.dump(dict_to_save, outfile)

print("1ere fin")


data_a_tokenizer=new_data.drop(['Nb_points', 'POLYLINE', 'TIMESTAMP'],axis=1)
dict_to_save2=data_a_tokenizer.to_dict()
with open("/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json", "w") as outfile:
    json.dump(dict_to_save2, outfile)

print("fin")


