import pandas as pd
import json
from transformers import BertModel, BertTokenizer

with open('/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json', 'r') as openfile:
    json_loaded = json.load(openfile)

data_clean = pd.DataFrame(data=json_loaded)


data_format=data_clean.copy()

data_format['INPUT']=data_format['Tokenization'][-1]



