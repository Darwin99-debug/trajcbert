from transformers import BertForSequenceClassification
import json
import os
from sklearn.model_selection import train_test_split
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import argparse


# DIR_INPUTS_IDS = (
#     "/home/daril/trajcbert/savings_for_parrallel_1_2/input_ids_f_833383.pkl"
# )
# DIR_ATTENTION_MASKS = (
#     "/home/daril/trajcbert/savings_for_parrallel_1_2/attention_masks_833383_opti.pkl"
# )
# DIR_TARGETS = "/home/daril/trajcbert/savings_for_parrallel_1_2/targets_833383_opti.pkl"
# PRETRAINED_MODEL_NAME = (
#     "/home/daril/trajcbert/savings_for_parrallel_1_2/model_before_training_opti_833383"
# )
# DATALOADER_DIR = "/home/daril/trajcbert/savings/test_dataloader_833383.pt"


def save_test_data_loader(
    input_ids_path,
    attention_masks_path,
    targets_path,
    dataloader_dir,
    batch_size=32,
):
    # load the lists saved in deb_train_gpu_parallel.py
    # the lists saved full_inputs, inputs_ids, attention_masks and the targets in different files /home/daril_kw/data/input_ids.pkl, /home/daril_kw/data/attention_masks.pkl, /home/daril_kw/data/targets.pkl

    with open(input_ids_path, "rb") as f:
        input_ids = pickle.load(f)

    with open(attention_masks_path, "rb") as f:
        attention_masks = pickle.load(f)

    with open(targets_path, "rb") as f:
        targets = pickle.load(f)

    targets_dict = {}
    # create a dictionary to convert the targets to numbers
    for i in range(len(targets)):
        if targets[i] not in targets_dict:
            targets_dict[targets[i]] = len(targets_dict)

    targets_input = [targets_dict[targets[i]] for i in range(len(targets))]

    train_data, test_data, train_targets, test_targets = train_test_split(
        input_ids, targets_input, random_state=2023, test_size=0.2
    )

    # the two _ are for test data and test targets

    train_masks, test_mask, _, _ = train_test_split(
        attention_masks, targets_input, random_state=2023, test_size=0.2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we create the dataloader for the test data

    test_inputs = torch.tensor(test_data)
    test_labels = torch.tensor(test_targets)
    test_masks = torch.tensor(test_mask)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(
        test_data
    )  # we don't use the DistributedSampler here because the validation is on a CPU
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    # save the test dataloader

    torch.save(test_dataloader, dataloader_dir)

    return test_dataloader


def main():
    # recover the parameters
    # --inputs_ids_path $DIR_INPUTS_IDS \
    # --attention_masks_path $DIR_ATTENTION_MASKS \
    # --targets_path $DIR_TARGETS \
    # --dataloader_dir $DATALOADER_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_ids_path", type=str, help="path to the inputs ids")
    parser.add_argument(
        "--attention_masks_path", type=str, help="path to the attention masks"
    )
    parser.add_argument("--targets_path", type=str, help="path to the targets")
    parser.add_argument("--dataloader_dir", type=str, help="path to the dataloader")
    args = parser.parse_args()
    DIR_INPUTS_IDS = args.inputs_ids_path
    DIR_ATTENTION_MASKS = args.attention_masks_path
    DIR_TARGETS = args.targets_path
    DATALOADER_DIR = args.dataloader_dir

    save_test_data_loader(
        DIR_INPUTS_IDS,
        DIR_ATTENTION_MASKS,
        DIR_TARGETS,
        DATALOADER_DIR,
        batch_size=32,
    )


