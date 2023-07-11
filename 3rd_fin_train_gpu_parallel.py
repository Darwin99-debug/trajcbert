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
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import h3
from sklearn.metrics import f1_score
from torch.utils.data.distributed import DistributedSampler
import argparse


WORLD_S=6


parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')






def main(rank, world_size):

    args = parser.parse_args()

    #local_rank = int(os.environ.get("SLURM_LOCALID")) 
    #rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    #current_device = local_rank
    current_device = rank
    torch.cuda.set_device(current_device)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
    """def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size) """   
    """
    def cleanup():
        
        
        dist.destroy_process_group()
    """
    def prepare(rank, world_size, batch_size, train_inputs, train_masks, train_labels, pin_memory=False, num_workers=0):
        dataset = TensorDataset(train_inputs, train_masks, train_labels)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
        return dataloader


    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))


    def flat_f1(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat,pred_flat,average='macro')

    print("imports done")

    #load the lists saved in deb_train_gpu_parallel.py
    # the lists saved full_inputs, inputs_ids, attention_masks and the targets in different files /home/daril_kw/data/input_ids.pkl, /home/daril_kw/data/attention_masks.pkl, /home/daril_kw/data/targets.pkl
    """
    with open('/home/daril_kw/data/input_ids.pkl', 'rb') as f:
        input_ids = pickle.load(f)

    with open('/home/daril_kw/data/attention_masks.pkl', 'rb') as f:
        attention_masks = pickle.load(f)

    with open('/home/daril_kw/data/targets.pkl', 'rb') as f:
        targets = pickle.load(f)

    with open('/home/daril_kw/data/full_inputs.pkl', 'rb') as f:
        full_inputs = pickle.load(f)

    print("gestion des targets")

    targets_dict={}
    for i in range(len(targets)):
        if targets[i] not in targets_dict:
            targets_dict[targets[i]]=len(targets_dict)

    targets_input=[targets_dict[targets[i]] for i in range(len(targets))]"""
    train_data, test_input, train_targets, test_targets = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2)
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_targets,random_state=2023, test_size=0.1)

    train_masks, test_masks, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(train_masks, train_targets,random_state=2023, test_size=0.1)

    print(" parallélisation : passage gpu")
#we go on the gpu
    device = torch.device("cuda")

    print("conversion des données (listes) en tenseurs")
#on convertit les données en tenseurs
    train_inputs = torch.tensor(train_inputs).to(rank)
    validation_inputs = torch.tensor(validation_inputs).to(rank)
    test_inputs = torch.tensor(test_input).to(rank)

    train_labels = torch.tensor(train_labels).to(rank)
    validation_labels = torch.tensor(validation_labels).to(rank)
    test_labels = torch.tensor(test_targets).to(rank)

    train_masks = torch.tensor(train_masks).to(rank)
    validation_masks = torch.tensor(validation_masks).to(rank)
    test_masks = torch.tensor(test_masks).to(rank)


    batch_size = 32

    


    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)

    prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data,sampler=prediction_sampler, batch_size=batch_size)

    seed_val = 2023
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    """setup(rank, world_size)"""
    
    

    train_dataloader = prepare(rank, world_size, batch_size, train_inputs, train_masks, train_labels)


    model = BertForSequenceClassification.from_pretrained("/home/daril_kw/data/model_before_training")
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)

# Number of training epochs. The BERT authors recommend between 2 and 4.
    epochs = 4

# Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)


#we store the loss and accuracy of each epoch
    loss_values = []
    accuracy_values = []
    f1_values = []



# For each epoch...
    for epoch_i in range(0, epochs):
        print("")
        train_dataloader.sampler.set_epoch(epoch_i)
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
    #cleanup
    dist.destroy_process_group()
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('/home/daril_kw/data/model_trained_parallel_gpu')

import torch.multiprocessing as mp
if __name__ == '__main__':
    world_size = WORLD_S
    with open('/home/daril_kw/data/input_ids.pkl', 'rb') as f:
        input_ids = pickle.load(f)

    with open('/home/daril_kw/data/attention_masks.pkl', 'rb') as f:
        attention_masks = pickle.load(f)

    with open('/home/daril_kw/data/targets.pkl', 'rb') as f:
        targets = pickle.load(f)

    with open('/home/daril_kw/data/full_inputs.pkl', 'rb') as f:
        full_inputs = pickle.load(f)

    print("gestion des targets")

    targets_dict={}
    for i in range(len(targets)):
        if targets[i] not in targets_dict:
            targets_dict[targets[i]]=len(targets_dict)

    targets_input=[targets_dict[targets[i]] for i in range(len(targets))]
    mp.spawn(main,args=(world_size,),nprocs=world_size)

