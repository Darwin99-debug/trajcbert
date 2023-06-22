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

import os 
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
