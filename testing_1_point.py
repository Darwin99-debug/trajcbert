

from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np 
import os
import json



DIR_MODEL_NOT_TRAINED = '/home/daril_kw/data/model_resized_embeddings_test'

with open("/home/daril_kw/trajcbert/trajcbert/config_test_gene.json") as json_file:
        config = json.loads(json_file.read())

epoch = config['num_epochs']


#we check if '/home/daril_kw/data/test/temp_file/checkpoint_epoch_{epoch}.pt' exists, otherwise we take the last checkpoint
while not os.path.exists('/home/daril_kw/data/test/temp_file/checkpoint_epoch_{epoch}.pt') and epoch > 0:
  epoch -= 1
   
TRAINED_MODEL_PATH = f"/home/daril_kw/data/test/temp_file/checkpoint_epoch_1.pt"
DIR_TEST_DATALOADER = '/home/daril_kw/data/test_dataloader_parallel_gene.pt'


device = torch.device("cpu")

#load the prediction_dataloader
#prediction_dataloader = torch.load('/home/daril_kw/data/pred_dataloader_v_small.pt')
prediction_dataloader = torch.load(DIR_TEST_DATALOADER)

# we load the model
model = BertForSequenceClassification.from_pretrained(DIR_MODEL_NOT_TRAINED)

state_dict = torch.load(TRAINED_MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

print("we evaluate")
model.eval()
 
# Tracking variables 
predictions , true_labels, list_inputs_test = [], [], []
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




#save flat_list_inputs_test
