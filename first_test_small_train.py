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
from transformers get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel


with open('/home/daril_kw/data/02.06.23/train_clean.json', 'r') as openfile:

    # Reading from json file
    json_loaded = json.load(openfile)

print("We put the data in a dataset.")
 
#we put them in a dataframe
data_format = pd.DataFrame(data=json_loaded)

#we keep only 60 rows
data_format = data_format[:60]

#we create the correct tokenization column
data_format['Tokenization_2'] = data_format['POLYLINE'].apply(lambda x: [h3.geo_to_h3(x[i][0], x[i][1], 10) for i in range(len(x))])

#on récupère la date à partir du timestamp
data_format['DATE'] = data_format['TIMESTAMP'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

#a partir de cela on récupère le jour de la semaine sous forme d'un nombre entre 1 et 7 pour lundi à dimanche
data_format['DAY'] = data_format['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[2]))

#ensuite, on récupère l'heure sous forme d'un nombre entre 0 et 23
data_format['HOUR'] = data_format['DATE'].apply(lambda x: x.split(' ')[1].split(':')[0])

#enfin on recupère le numéro de la semaine dans l'année
data_format['WEEK'] = data_format['DATE'].apply(lambda x: str(datetime.datetime.strptime(x.split(' ')[0],'%Y-%m-%d').isocalendar()[1]))



#we remove the useless columns
data_format.drop(['MISSING_DATA','DATE','ORIGIN_CALL','TRIP_ID', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'Nb_points', 'TIMESTAMP' ],axis=1,inplace=True)



#we put the data in a str format
print("we put in the right format")
data_format['CALL_TYPE'] = data_format['CALL_TYPE'].apply(lambda x: str(1) if x=='A' else str(2) if x=='B' else str(3))
if type(data_format['TAXI_ID'][0])!=str:
    data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: str(x))
#idem pour le call_type
if type(data_format['CALL_TYPE'][0])!=str:
    data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: str(x))

print("gestion du tokenizer commencée")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


liste_token_geo = []

for i in range(len(data_format)):
    for j in range(len(data_format['Tokenization_2'][i])):
        liste_token_geo.append(data_format['Tokenization_2'][i][j])

#on enlève les doublons
liste_token_geo = list(set(liste_token_geo))

#on garde le nombre de tokens géographiques pour la suite
nb_token_geo = len(liste_token_geo)

#On ajoute les tokens géographiques au tokenizer
tokenizer.add_tokens(liste_token_geo)
print("On a le tokenizer final")

#On a besoin du nombre de labels, celui-ci correspond au nombre de tokens géographiques + 1 (pour le token [SEQ] indiquant la fin de la séquence)

nb_labels = nb_token_geo + 1

model=BertForSequenceClassification.from_pretrained("bert-base-cased",num_labels=nb_labels)
#on adapte la taille de l'embedding pour qu'elle corresponde au nombre de tokens géographiques + 1
model.resize_token_embeddings(len(tokenizer))


print("gestion du format de l'input commencée")
#gestion du format de l'input
data_format['HOUR']=data_format['HOUR'].apply(lambda x: ' '+x)
data_format['WEEK']=data_format['WEEK'].apply(lambda x: ' '+x)
data_format['CALL_TYPE']=data_format['CALL_TYPE'].apply(lambda x: ' '+x)
data_format['TAXI_ID']=data_format['TAXI_ID'].apply(lambda x: ' '+x)
data_format['DAY']=data_format['DAY'].apply(lambda x: ' '+x)

data_format['CONTEXT_INPUT'] =data_format['Tokenization_2'].apply(lambda x: x[-1]) + data_format['DAY'] + data_format['HOUR'] + data_format['WEEK'] + data_format['CALL_TYPE'] + data_format['TAXI_ID']

len_context_info = len(data_format['CONTEXT_INPUT'][0].split(' '))

#la colonne DEB_TRAJ sera la colonne Tokenization jusqu'a l'avant-dernier token exclu

data_format['DEB_TRAJ']=data_format['Tokenization_2'].apply(lambda x: x[:-2])


# on gère la longueur de la colonne CONTEXT_INPUT pour qu'après la concaténation, elle ne dépasse pas 512 tokens
#le -2 correspond aux deux tokens spéciaux [CLS] et [SEP]
# for exemple here, if the trajectory input is too long, we keep the 512-6-2=504 last tokens
data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: x[-(512-len_context_info-2):] if len(x)>512-len_context_info-2 else x)

#then we keep the column in form of a string
data_format['DEB_TRAJ']=data_format['DEB_TRAJ'].apply(lambda x: ' '.join(x))

data_format['TARGET']=data_format['Tokenization_2'].apply(lambda x: x[-2])


#on enlève les colonnes inutiles
if 'Tokenization' in data_format.columns:
    data_format.drop(['Tokenization'],axis=1,inplace=True)
if 'CALL_TYPE' in data_format.columns:
    data_format.drop(['CALL_TYPE'],axis=1,inplace=True)
if 'TAXI_ID' in data_format.columns:
    data_format.drop(['TAXI_ID'],axis=1,inplace=True)
if 'DAY' in data_format.columns:
    data_format.drop(['DAY'],axis=1,inplace=True)
if 'HOUR' in data_format.columns:
    data_format.drop(['HOUR'],axis=1,inplace=True)
if 'WEEK' in data_format.columns:
    data_format.drop(['WEEK'],axis=1,inplace=True)
if 'Nb_points_token' in data_format.columns:
    data_format.drop(['Nb_points_token'],axis=1,inplace=True)


c_inputs=data_format.CONTEXT_INPUT.values
traj_inputs=data_format.DEB_TRAJ.values
targets=data_format.TARGET.values

print("concaténation des inputs, padding etc")

input_ids = []
full_inputs = []
attention_masks = []
for i in tqdm(range(len(c_inputs))):
    #no truncation is needed because we managed it before

    #we concatenate the context input and the trajectory input adding manually the CLS token and the SEP token
    full_input = '[CLS] ' + c_inputs[i] + ' ' + traj_inputs[i] + ' [SEP]'
    full_inputs.append(full_input)
    #encoded_c_input=tokenizer.encode(c_inputs[i], add_special_tokens=False)
    #encoded_traj_input=tokenizer.encode(traj_inputs[i], add_special_tokens=False)
    #we add manually the CLS token and the SEP token when we concatenate the two inputs
    #encoded_full_input=[101] + encoded_c_input + encoded_traj_input + [102]
    #the[101] token is the CLS token and the [102] token is the SEP token

    encoded_full_input=tokenizer.encode(full_input, add_special_tokens=False)
    #we pad the input to the maximum length of 512
    encoded_full_input=encoded_full_input + [0]*(512-len(encoded_full_input))
    input_ids.append(encoded_full_input)
    #we create the attention mask
    att_mask = [float(i>0) for i in encoded_full_input]
    attention_masks.append(att_mask)
    #the attention mask is a list of 0 and 1, 0 for the padded tokens and 1 for the other tokens
    #the float(i>0) is 0 if i=0 (ie if the token is a padded token) and 1 if i>0 (ie if the token is not a padded token)



#gestion des targets
targets_dict={}
for i in range(len(targets)):
    if targets[i] not in targets_dict:
        targets_dict[targets[i]]=len(targets_dict)

targets_input=[targets_dict[targets[i]] for i in range(len(targets))]

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


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#we go on the cpu
device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("/home/daril_kw/data/model_final",num_labels=nb_labels)
model.to(device)
#model = DistributedDataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)

# Number of training epochs. The BERT authors recommend between 2 and 4.
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)


#on définit les fonctions utiles pour l'entrainement


#la focntion
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


seed_val = 2023
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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
        

        """
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)"""

        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch


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




#we save the model
output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
#we save the loss and accuracy values
np.save(output_dir+'loss_values.npy',loss_values)
np.save(output_dir+'accuracy_values.npy',accuracy_values)