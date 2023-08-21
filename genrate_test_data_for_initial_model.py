from transformers import  BertForSequenceClassification
import json
import os
from sklearn.model_selection import train_test_split
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler


DIR_INPUTS_IDS = '/home/daril/trajcbert/savings_for_parrallel_1_2/input_ids_f_833383.pkl'
DIR_ATTENTION_MASKS = '/home/daril/trajcbert/savings_for_parrallel_1_2/attention_masks_833383_opti.pkl'
DIR_TARGETS = '/home/daril/trajcbert/savings_for_parrallel_1_2/targets_833383_opti.pkl'
PRETRAINED_MODEL_NAME = '/home/daril/trajcbert/savings_for_parrallel_1_2/model_before_training_opti_833383'
DATALOADER_DIR = '/home/daril/trajcbert/savings/test_dataloader_833383.pt'











def load_data(batch_size=32):
     #load the lists saved in deb_train_gpu_parallel.py
    # the lists saved full_inputs, inputs_ids, attention_masks and the targets in different files /home/daril_kw/data/input_ids.pkl, /home/daril_kw/data/attention_masks.pkl, /home/daril_kw/data/targets.pkl


    with open(DIR_INPUTS_IDS, 'rb') as f:
        input_ids = pickle.load(f)

    with open(DIR_ATTENTION_MASKS, 'rb') as f:
        attention_masks = pickle.load(f)

    with open(DIR_TARGETS, 'rb') as f:
        targets = pickle.load(f)


    targets_dict={}
    # create a dictionary to convert the targets to numbers
    for i in range(len(targets)):
        if targets[i] not in targets_dict:
            targets_dict[targets[i]]=len(targets_dict)

    targets_input=[targets_dict[targets[i]] for i in range(len(targets))]

    train_data, test_data, train_targets, test_targets = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2)

    # the two _ are for test data and test targets
    
    train_masks, test_mask, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we create the dataloader for the test data


    
   

   

    test_inputs = torch.tensor(test_data)
    test_labels = torch.tensor(test_targets)
    test_masks = torch.tensor(test_mask)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data) # we don't use the DistributedSampler here because the validation is on a CPU
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    # save the test dataloader
  

    torch.save(test_dataloader, DATALOADER_DIR)


    return test_dataloader


load_data()