import pandas as pd
import json
from transformers import BertModel, BertTokenizer

with open('/home/daril_kw/data/dataset_to_tokenize_wo_timestamp.json', 'r') as openfile:
    json_loaded = json.load(openfile)



