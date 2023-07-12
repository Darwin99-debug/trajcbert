from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
import random
import json
import time
import os
import datetime
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel

#load the tokenizer from /home/daril_kw/data/tokenizer_final
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_final')
#load the dataset from home/daril_kw/data/data_with_time_info_ok.json
with open('/home/daril_kw/data//home/daril_kw/data/data_with_time_info_ok.json', 'r') as openfile:
    json_loaded = json.load(openfile)
    
data_format = pd.DataFrame(data=json_loaded)