#imports

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import json
from datasets import load_dataset

#we load the json file
with open('/home/daril_kw/data/data_formated_final.json', 'r') as openfile:
    json_loaded = json.load(openfile)
data_format = pd.DataFrame(data=json_loaded)


#we create a dataset from the file data_formated.json
class TrajDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        data=json.load(open(json_file))
        self.tokenizer = tokenizer
        self.input = data['INPUT']
        self.target = data['TARGET']    

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        return input, target    
    
#we load the tokenizer
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/trajcbert/trajcbert/new_tokenizer', sep_token='[SEP]', cls_token='[CLS]')

def preprocess_function(dataloader):
    return tokenizer(dataloader["INPUT"], padding=True, max_length=512)


data = TrajDataset(data_format['INPUT'], data_format['TARGET'])

tokenized_data = data.map(preprocess_function, batched=True)



from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)




import evaluate

accuracy = evaluate.load("accuracy")


import numpy as np


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)

#we create a map of the expected id to their token thanks to the tokenizer

#from the id to the token
id2token = tokenizer.ids_to_tokens

#from the token to the id
token2id= tokenizer.vocab



from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


model = AutoModelForSequenceClassification.from_pretrained(

    "distilbert-base-uncased", num_labels=len(tokenizer), id2token=id2label, 2id=label2id

)