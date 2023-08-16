

from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler








DIR_INPUTS_IDS = '/home/daril/trajcbert/savings_for_parrallel_1_2/input_ids_f_833383.pkl'
DIR_ATTENTION_MASKS = '/home/daril/trajcbert/savings_for_parrallel_1_2/attention_masks_833383_opti.pkl'
DIR_TARGETS = '/home/daril/trajcbert/savings_for_parrallel_1_2/targets_833383_opti.pkl'
PRETRAINED_MODEL_NAME = '/home/daril/trajcbert/savings_for_parrallel_1_2/model_before_training_opti_833383'









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


    
   

   

    test_inputs = torch.tensor(test_data)
    test_labels = torch.tensor(test_targets)
    test_masks = torch.tensor(test_mask)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data) # we don't use the DistributedSampler here because the validation is on a CPU
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
    






    return test_dataloader

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






#load the prediction_dataloader
prediction_dataloader = torch.load('models/pred_dataloader_v_small.pt')



# we load the model
model = BertForSequenceClassification.from_pretrained('models/model_trained_cpu_version')
print("we evaluate")
model.eval()
 
# Tracking variables 
predictions , true_labels, list_inputs_test = [], [], []

# losses
losses = 0
print("We predict")
# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
   
  # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]
    losses += outputs[0].mean().item()

  # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

  #Store the inputs

    list_inputs_test.append(b_input_ids.tolist())

print('    DONE.')




matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

pred_label= []
# compute the loss

# For each input batch...
for i in range(len(true_labels)):

  # The predictions for this batch are a 2-column ndarray (one column for "0"
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  pred_label.append(pred_labels_i)
  # Calculate and store the coef for this batch.
  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
  matthews_set.append(matthews)


# Combine the predictions for each batch into a single list of 0s and 1s.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = [item for sublist in true_labels for item in sublist]


# Combine the inputs for each batch into a single list.
flat_list_inputs_test = [item for sublist in list_inputs_test for item in sublist]

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('MCC: %.3f' % mcc)


# compute the accuracy
accuracy = (flat_true_labels == flat_predictions).mean()
print('accuracy: %.3f' % accuracy)

# print the loss
print('loss: %.3f' % (losses/len(true_labels)))





#save flat_list_inputs_test
