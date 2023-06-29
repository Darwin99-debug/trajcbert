import pickle
with open('/home/daril_kw/data/input_ids_small.pkl', 'rb') as f:
    input_ids = pickle.load(f)
with open('/home/daril_kw/data/attention_masks_small.pkl', 'rb') as f:
    attention_masks = pickle.load(f)
with open('/home/daril_kw/data/targets_encoded_small.pkl', 'rb') as f:
    targets_encoded = pickle.load(f)

input_ids =input_ids[:60]
attention_masks = attention_masks[:60]
targets_encoded=targets_encoded[:60]
targets_matching=targets_encoded.copy()
targets_matching.sort()
 

with open('/home/daril_kw/data/targets_matching_small.pkl', 'wb') as f:
    pickle.dump(targets_matching, f)

targets_input=[]
for i in range(len(targets_encoded)):
	targets_input.append(targets_matching.index(targets_encoded[i]))

with open('/home/daril_kw/data/targets_input_small.pkl', 'wb') as f:
    pickle.dump(targets_input, f)






#we separate the data into test train and validation sets
from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2) 
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_targets,random_state=2023, test_size=0.1)

train_masks, test_masks, _, _ = train_test_split(attention_masks, targets_encoded,random_state=2023, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(train_masks, train_targets,random_state=2023, test_size=0.1)


#print("we start on the CPU")
print("we go on the GPU")
import torch
device = torch.device("cuda")
#device = torch.device("cpu")

print("Loading of the tokenizer")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_full_2')


train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
test_inputs = torch.tensor(test_data)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_targets)

#same for the masks

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)


print("We have tensors as wanted")

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32

# Create the DataLoader for our training set.

#the line below is to create a tuple of tensors
train_data = TensorDataset(train_inputs, train_masks, train_labels)

#the line below is to create a sampler to sample the data during training ie to shuffle the data
train_sampler = RandomSampler(train_data)

#the line below is to create the dataloader which is actually the object that will be used for training.
#This object works as a generator, it will generate the data in the form of batches of size batch_size
train_dataloader = DataLoader(train_data,sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
#we do the same for the validation set
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)

print("The data loading is finished")

from transformers import BertForSequenceClassification

from torch.nn.parallel import DistributedDataParallel


with open('/home/daril_kw/data/list_token_geo2.txt','r') as f:
    list_geographic_token=f.readlines()
nb_labels= len(list_geographic_token)+2
#the +1 is for the end token []
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=nb_labels,output_attentions = False,output_hidden_states = False)
#we change the embedding size to match the tokenizer
model.resize_token_embeddings(len(tokenizer))


model.save_pretrained('/home/daril_kw/data/model_bert_classification_2')

model.to(device)
from torch.nn.parallel import DataParallel
model = DistributedDataParallel(model)

print("The model is loaded")

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



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
