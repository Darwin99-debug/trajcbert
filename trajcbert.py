import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm



trajcbert = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/trajcbert/trajcbert/new_tokenizer')


