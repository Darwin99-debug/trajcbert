#importation des librairies

import pandas as pd
import json
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.parallel import DataParallel

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")





#we load the json file
with open('/home/daril_kw/data/dataset_version_timestamp_info_extracted.json', 'r') as openfile:
    json_loaded = json.load(openfile)
data_format = pd.DataFrame(data=json_loaded)


#on vérifie que les données dans la colonne Tokenization sont bien des listes
data_format['Tokenization'].apply(lambda x: isinstance(x,list)).value_counts()
print("Verification du format des données effectuée.")
#on va mettre dans une nouvelle colonne INPUT la concaténation dans cet ordre, du jetpn CLS, du dernier élément de ligne sur la colonne Tokenization, la colonne CALL_TYPE, la colonne TAXI_ID,  et enfin la liste des éléments la ligne pour la colonne Tokenization jusqu'à l'avant avant dernier élément


#on commence par transformer les éléments de la colonne TAXI_ID en string si ils ne le sont pas déjà
if type(data_format['TAXI_ID'][0])!=str:
    data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))
#idem pour le call_type
if type(data_format['CALL_TYPE'][0])!=str:
    data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: str(x))

#on continue en ajoutant un espace devant les éléments de la colonne HOUR, WEEK, CALL_TYPE, TAXI_ID en prévision de la concaténation pour séparer d'un espace les éléments de la colonne Tokenization et les autres colonnes
data_format['HOUR']=data_format['HOUR'].apply(lambda x: ' '+x)
data_format['WEEK']=data_format['WEEK'].apply(lambda x: ' '+x)
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)
data_format['DAY']=data_format['DAY'].apply(lambda x: ' '+x)

# la colonne CONTEXT_INPUT sera la concaténation du jour de la semaine, de l'heure et de la semaien de l'année pui de la colonne CALL_TYPE, de la colonne TAXI_ID, d'un espace et du dernier token de la colonne Tokenization
data_format['CONTEXT_INPUT'] =data_format['Tokenization'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']


#la colonne DEB_TRAJ sera la colonne Tokenization jusqu'a l'avant-dernier token exclu
data_format['DEB_TRAJ']=data_format['Tokenization'].apply(lambda x: x[:-2])
#we truncate the beggining of the trajectory input if it is too long to fit in the 512 tokens after the concatenation
#we keep the end of the trajectory

len_context_info =6
#the 2 is for the CLS and SEP tokens
    data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: x[-(512-len_context_info-2):])
#then we keep the column in form of a string
data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: ' '.join(x))




# la target sera l'avant dernier élément de la colonne Tokenization
data_format['TARGET']=data_format['Tokenization'].apply(lambda x: x[-2])

#on supprime les colonnes inutiles si elles existent encore
if 'Tokenization' in data_format.columns:
    data_format.drop(['Tokenization'],axis=1,inplace=True)
#if 'CALL_TYPE' in data_format.columns:
 #   data_format.drop(['CALL_TYPE'],axis=1,inplace=True)
#if 'TAXI_ID' in data_format.columns:
 #   data_format.drop(['TAXI_ID'],axis=1,inplace=True)
 data_format.drop(['Nb_points_token'],axis=1,inplace=True)
    

#on sauvegarde le dataframe dans un fichier json
data_format.to_json('/home/daril_kw/data/data_formated_with_time_info.json',orient='records')



#A présent, on va gérer le format des données en entrée (dataloader)
#on load le tokenize

tokenizer = BertTokenizer.from_pretrained('/home/daril_kw/data/tokenizer_full')

c_inputs=data_format.CONTEXT_INPUT.values
traj_inputs=data_format.DEB_TRAJ.values
targets=data_format.TARGET.values

from tqdm import tqdm

input_ids = []
full_inputs = []
for i in tqdm(range(len(c_inputs))):
    #no truncation is needed because we managed it before

    #we concatenate the context input and the trajectory input adding manually the CLS token and the SEP token
    full_inputs.append('[CLS] ' + c_inputs[i] + ' ' + traj_inputs[i] + ' [SEP]')

    encoded_c_input=tokenizer.encode(c_inputs[i], traj_inputs[i], add_special_tokens=False)
    encoded_traj_input=tokenizer.encode(traj_inputs[i], add_special_tokens=False)



    #we add manually the CLS token and the SEP token when we concatenate the two inputs
    encoded_full_input=[101] + encoded_c_input + encoded_traj_input + [102]
    #the[101] token is the CLS token and the [102] token is the SEP token
    #we pad the input to the maximum length of 512
    encoded_full_input=encoded_full_input + [0]*(512-len(encoded_full_input))
    input_ids.append(encoded_full_input)

print('Original: ', full_inputs[0])
print('Token IDs:', input_ids[0])


# Create attention masks
attention_masks = []

for seq in input_ids:
    att_mask = [float(i>0) for i in seq]
    attention_masks.append(att_mask)

    
# Use train_test_split to split our data into train and validation sets for training
from sklearn.model_selection import train_test_split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, targets,random_state=2023, test_size=0.1)
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, targets,random_state=2023, test_size=0.1)


# Convert all inputs and labels into torch tensors, the required datatype for our model.
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#once we have everything we want, we manage the dataloader
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


#we load the model for classification we saved in another file

#for that we need the number of labels ie the number of geographic token we have in the file list_geographic_token.txt
#we load the file
with open('/home/daril_kw/data/list_geographic_token.txt','r') as f:
    list_geographic_token=f.readlines()
nb_labels= len(list_geographic_token)
model = BertForSequenceClassification.from_pretrained('/home/daril_kw/data/model_classif_full',num_labels=nb_labels)


# Tell pytorch to run this model on the GPU.
model.to(device)

# tell pytorch to run this model on multiple GPUs
model = DataParallel(model)



# Get all of the model's parameters as a list of tuples.
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



#we define the optimizer
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)

# Number of training epochs. The BERT authors recommend between 2 and 4.
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

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

#trainning loop
#we store the loss and accuracy of each epoch
loss_values = []
accuracy_values = []
f1_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader),elapsed))
        #we unpack the batch    
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        #we set the gradients to zero
        model.zero_grad()
        #we make the forward pass
        outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
        #we get the loss
        loss = outputs[0]
        #we accumulate the loss
        total_loss += loss.item()
        #we make the backward pass
        loss.backward()
        #we clip the gradient to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #we update the parameters
        optimizer.step()
        #we update the learning rate
        scheduler.step()
    # Calculate the average loss over all of the batches.  
    avg_train_loss = total_loss / len(train_dataloader)
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Running Validation...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
    model.eval()
    # Tracking variables
    eval_loss, eval_accuracy,eval_f1 = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        #we unpack the batch
        b_input_ids, b_input_mask, b_labels = batch
        #we don't compute the gradient
        with torch.no_grad():
            #we make the forward pass
            outputs = model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask)
            #we get the logits
            logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        #we compute the accuracy
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #we compute the f1 score
        tmp_eval_f1 = flat_f1(logits, label_ids)
        #we accumulate the accuracy
        eval_accuracy += tmp_eval_accuracy
        #we accumulate the f1 score
        eval_f1 += tmp_eval_f1
        #we accumulate the number of examples
        nb_eval_examples += b_input_ids.size(0)
        #we accumulate the number of steps
        nb_eval_steps += 1
    #we compute the accuracy
    eval_accuracy = eval_accuracy / nb_eval_examples
    #we compute the f1 score
    eval_f1 = eval_f1 / nb_eval_examples
    print("  Accuracy: {0:.2f}".format(eval_accuracy))
    print("  F1 score: {0:.2f}".format(eval_f1))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    #we store the accuracy
    accuracy_values.append(eval_accuracy)
    #we store the f1 score
    f1_values.append(eval_f1)
print("")
print("Training complete!")

#in the trainning loop, the loss is computed for each batch, so we have to compute the average loss for each epoch
#the loss function is the cross entropy loss that is to say the negative log likelihood loss
