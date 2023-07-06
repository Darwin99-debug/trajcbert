
#we go on the gpu
device = torch.device("cuda")

torch.cuda.set_device(0)
torch.cuda.set_device(1)

world_size=2

dist.init_process_group("nccl", world_size=world_size)




# Create the DataLoader for our training set, one for validation set and one for test set


validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)

prediction_data = TensorDataset(test_inputs, test_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data,sampler=prediction_sampler, batch_size=batch_size)


#model = BertForSequenceClassification.from_pretrained("/home/daril_kw/data/model_final",num_labels=nb_labels)
model.to(device)
#model = DistributedDataParallel(model)


#on définit les fonctions utiles pour l'entrainement


#la focntion
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

seed_val = 2023
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def main(rank, world_size):
    
    dataset = TensorDataset(train_inputs, train_masks, train_labels)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=False, shuffle=False, sampler=sampler)

    # prepare the dataloader
    train_dataloader = dataloader
    model = Model().to(rank)
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
        dataloader.sampler.set_epoch(epoch_i)
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


import torch.multiprocessing as mp
if __name__ == '__main__':
    world_size = 2
    mp.spawn(main,args=(world_size),nprocs=world_size)


"""

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
np.save(output_dir+'accuracy_values.npy',accuracy_values)"""


model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained('/home/daril_kw/data/model_trained')

np.save('/home/daril_kw/data/model_trained/loss_values.npy',loss_values)
np.save('/home/daril_kw/data/model_trained/accuracy_values.npy',accuracy_values)
