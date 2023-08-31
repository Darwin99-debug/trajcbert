from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import argparse


# This file test the first version of the model: classification with context


# PRETRAINED_MODEL_NAME = "/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version_full_bs_32_20_epochs_with_context"
# DATALOADER_DIR = "/home/daril/trajcbert/savings/test_dataloader_833383.pt"


def main():
    # --pretrained_model_name $PRETRAINED_MODEL_NAME \
    # --dataloader_dir $DATALOADER_DIR


    # recovery the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name",
        default="/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version_full_bs_32_20_epochs_with_context",
        type=str,
        help="The pretrained model name",
    )
    parser.add_argument(
        "--dataloader_dir",
        default="/home/daril/trajcbert/savings/test_dataloader_833383.pt",
        type=str,
        help="The dataloader directory",
    )

    args = parser.parse_args()

    PRETRAINED_MODEL_NAME = args.pretrained_model_name
    DATALOADER_DIR = args.dataloader_dir

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the prediction_dataloader
    prediction_dataloader = torch.load(DATALOADER_DIR)

    # we load the model
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
    model.to(device)
    print("we evaluate")
    model.eval()

    # Tracking variables
    predictions, true_labels, list_inputs_test = [], [], []

    # losses
    losses = 0
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
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        logits = outputs[0]
        losses += outputs[0].mean().item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        # Store predictions and true labels
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

    # save flat_list_inputs_test


if __name__ == "__main__":
    main()
