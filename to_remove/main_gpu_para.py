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

def training_loop(rank, world_size):
    setup(rank, world_size)
    # prepare the dataloader
    train_dataloader = prepare(rank, world_size)
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


def main():
    world_size = 2
    torch.distributed.launch(training_loop,args=(world_size,),nprocs=world_size,join=True)


