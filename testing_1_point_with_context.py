from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler




# This file test the first version of the model: classification with context





DIR_INPUTS_IDS = '/home/daril_kw/data/savings_for_parallel_60/input_ids_full_opti_for_para_60.pkl'
DIR_ATTENTION_MASKS = '/home/daril_kw/data/savings_for_parallel_60/attention_masks_full_opti_for_para_60.pkl'
DIR_TARGETS = '/home/daril_kw/data/savings_for_parallel_60/targets_full_opti_for_para_60.pkl'
PRETRAINED_MODEL_NAME = 'bert-base-cased'










# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# load the prediction_dataloader
prediction_dataloader = DataLoader(
    TensorDataset(torch.load(DIR_INPUTS_IDS), torch.load(DIR_ATTENTION_MASKS), torch.load(DIR_TARGETS)),
    sampler=SequentialSampler(TensorDataset(torch.load(DIR_INPUTS_IDS), torch.load(DIR_ATTENTION_MASKS), torch.load(DIR_TARGETS))),
    batch_size=32
    


)


# we load the model
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
model.to(device)
print("we evaluate")
model.eval()

# Tracking variables
predictions, true_labels, list_inputs_test = [], [], []

# losses
losses = 0

losess_witouht_mean = 0
print("We predict")
# Predict
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # move to device
    b_input_ids = b_input_ids.to(device)
    b_input_mask = b_input_mask.to(device)
    b_labels = b_labels.to(device)

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        # the ouputs are a tuple with the loss and the logits
        # the losses are the item 0 of the tuple
        # and the logits are the item 1 of the tuple
        # The loss is computed with the CrossEntropyLoss

    logits = outputs[0]
    print(f"logits: {logits}\n logits shape: {logits.shape} \n losses: {outputs[0]}  \n outputs shape: {outputs[0].shape}")
    losses += outputs[0].mean().item()
    losess_witouht_mean += outputs[0].item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to("cpu").numpy()

    # Store predictions and true labels
    # we have to append  the max of the logits
    # because the logits are the output of the softmax
    # and the max of the logits is the class with the highest probability
    predictions.append(logits)
    true_labels.append(label_ids)

    # Store the inputs

    list_inputs_test.append(b_input_ids.tolist())

print("DONE.")


matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print("Calculating Matthews Corr. Coef. for each batch...")

pred_label = []
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

print("MCC: %.3f" % mcc)


# compute the accuracy
accuracy = (flat_true_labels == flat_predictions).mean()
print("accuracy: %.3f" % accuracy)

# print the loss
print("loss: %.3f" % (losses / len(true_labels)))

print("loss without mean: %.3f" % (losess_witouht_mean / len(true_labels)))


# save flat_list_inputs_test
