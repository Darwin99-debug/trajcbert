import pickle
with open('/home/daril_kw/data/input_ids_small.pkl', 'rb') as f:
    input_ids = pickle.load(f)
with open('/home/daril_kw/data/attention_masks_small.pkl', 'rb') as f:
    attention_masks = pickle.load(f)
with open('/home/daril_kw/data/targets_small.pkl', 'rb') as f:
    targets = pickle.load(f)


input_ids =input_ids[:60]
attention_masks = attention_masks[:60]
targets=targets[:60]


targets_dict={}
for i in range(len(targets)):
    if targets[i] not in targets_dict:
        targets_dict[targets[i]]=len(targets_dict)


targets_input=[targets_dict[targets[i]] for i in range(len(targets))]

#print("we start on the CPU")
print("we go on the GPU")
import torch
#device = torch.device("cuda")
device = torch.device("cpu")

from sklearn.model_selection import train_test_split

train_data, test_input, train_targets, test_targets = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2)
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_targets,random_state=2023, test_size=0.1)

train_masks, test_masks, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.2)
#train_masks, test_masks, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(train_masks, train_targets,random_state=2023, test_size=0.1)


#on convertit les données en tenseurs
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
test_inputs = torch.tensor(test_input)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_targets)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)




batch_size = 32

# Create the DataLoader for our training set, one for validation set and one for test set

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)

prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data,sampler=prediction_sampler, batch_size=batch_size)


with open('/home/daril_kw/data/list_token_geo2.txt','r') as f:
    list_geographic_token=f.readlines()
nb_labels= len(list_geographic_token)+2

model= BertForSequenceClassification.from_pretrained('/home/daril_kw/data/model_bert_classification_2', num_labels=nb_labels,output_attentions = False,output_hidden_states=False)


model.to(device)
from torch.nn.parallel import DataParallel
#model = DistributedDataParallel(model)

print("The model is loaded")


optimizer = torch.optim.AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)

# Number of training epochs. The BERT authors recommend between 2 and 4.
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

from transformers import get_linear_schedule_with_warmup

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)




# we define the function to calculate the accuracy of our predictions vs labels
import numpy as np
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#we define the function to calculate the time of execution of a epoch
import time
import datetime
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


#we define the function to calculate the f1 score of our predictions vs labels
from sklearn.metrics import f1_score
def flat_f1(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat,pred_flat,average='macro')


# Set the seed value all over the place to make this reproducible.
import random

seed_val = 2023
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print("Finished !")
