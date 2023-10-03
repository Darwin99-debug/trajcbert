from transformers import  BertForSequenceClassification
import json
import os
from sklearn.model_selection import train_test_split
import torch
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler


from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

DIR_INPUTS_IDS = '/home/daril/trajcbert/savings_for_parallel_computing_full/input_ids_full_opti.pkl'
DIR_ATTENTION_MASKS = '/home/daril/trajcbert/savings_for_parallel_computing_full/attention_masks_full_opti.pkl'
DIR_TARGETS = '/home/daril/trajcbert/savings_for_parallel_computing_full/targets_full_opti.pkl'
PRETRAINED_MODEL_NAME = '/home/daril/trajcbert/savings_for_parallel_computing_full/model_before_training_opti_full'

# WORLD_S=2


# parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
# parser.add_argument('--lr', default=0.1, help='')
# parser.add_argument('--batch_size', type=int, default=768, help='')
# parser.add_argument('--max_epochs', type=int, default=4, help='')
# parser.add_argument('--num_workers', type=int, default=0, help='')

# parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
# parser.add_argument('--dist-backend', default='nccl', type=str, help='')
# parser.add_argument('--world_size', default=1, type=int, help='')
# parser.add_argument('--distributed', action='store_true', help='')





 
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes: it is the number of available GPUs
    """
    os.environ["MASTER_ADDR"] = "localhost" #ip address of master node
    os.environ["MASTER_PORT"] = "12355"      #port of master node, this port is used by master node to listen to workers (can be any free port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # initialize the process group
    torch.cuda.set_device(rank) # define the current GPU as the one to use

def prepare_dataloader(dataset: Dataset,batch_size: int):
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=False, # Must be False when using DistributedSampler beacause it is already shuffled
                    #   pin_memory=True, # Automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
                      sampler = DistributedSampler(dataset) # Select a subset of the dataset (only works if shuffle=False)
      
                    )

def load_bert_model_and_tokenizer():
    """
    Load the pretrained BERT model and the tokenizer
    """
    #load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained('/home/daril/trajcbert/BERT_MODEL/tokenizer_augmented_full')

    #load the model
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)

    optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)
    return model, optimizer

class Trainer:


    """
    model: the model to train like BertForSequenceClassification
    train_data: the training data
    optimizer: the optimizer to use like AdamW
    gpu_id: the id of the gpu to use
    save_every: the number of epochs between each checkpoint
    scheduler: the scheduler to use like get_linear_schedule_with_warmup

    return: None

    """
    def __init__( 
        self,
        model: torch.nn.Module, 
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int, # the id of the gpu to use
        save_every: int, 
        scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.scheduler = scheduler
        self.model = DDP(model, device_ids=[gpu_id], )


    def _run_batch(self,  b_input_ids,b_input_mask,b_labels): 
        #we set the gradients to zero
        self.model.zero_grad()
        #we make the forward pass
        outputs = self.model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
        # the output is a tuple containing the loss and the logits (the output of the last layer)
        #we get the loss
        loss = outputs[0]
        #we make the backward pass
        loss.backward()
        #we clip the gradient to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        #we update the parameters
        self.optimizer.step()
        #we update the learning rate
        self.scheduler.step()
        #we accumulate the loss
        return loss.item()
       

    def _run_epoch(self, epoch):
        # we recover the batch size
        b_sz = len(next(iter(self.train_data))[0]) # batch size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch in self.train_data:
            batch = tuple(t.to(self.gpu_id) for t in batch)
            #unpack the batch
            input_ids, attention_mask, labels = batch
            # input_ids = input_ids.to(self.gpu_id)
            # attention_mask = batch["attention_mask"].to(self.gpu_id)
            # labels = batch["labels"].to(self.gpu_id)
            loss = self._run_batch(input_ids,attention_mask,labels)
            total_loss += loss
            # free memory
            del input_ids
            del attention_mask
            del labels
            # free GPU memory
            torch.cuda.empty_cache()
        average_loss = total_loss / len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average loss: {average_loss}")
        return average_loss
    def _accuracy(self, logits : np.ndarray, labels: np.ndarray) -> float:
        predicted = np.argmax(logits, axis=1).flatten()
        labels = labels.cpu().numpy()  # Convert torch.Tensor to numpy array
        correct = (predicted == labels)
        print(f"correct type: {type(correct)}  predited type: {type(predicted)}  labels type: {type(labels)}")
        correct = correct.sum()
        total = labels.size
        accuracy = correct / total
        return accuracy

    def _validate(self):
        model = self.model.module
        model.eval()
        #self.model.eval()
        eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        with torch.no_grad():
            for batch in self.validation_data:
                batch = tuple(t.to(self.gpu_id) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_input_ids = b_input_ids.to(self.gpu_id)
                b_input_mask = b_input_mask.to(self.gpu_id)
                b_labels = b_labels.to(self.gpu_id)

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                #outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
            
                

                eval_loss += loss.item()
    

                #logits = logits.detach().cpu().numpy() # logits is a tensor on the GPU, we need to move it to the CPU and then to the memory
              
                #label_ids = b_labels.to('cpu').numpy() # same for the labels
                #tmp_eval_accuracy = self._accuracy(logits, b_labels)
                #tmp_eval_f1 = f1_score(label_ids, np.argmax(logits, axis=1), average='macro')

                #eval_accuracy += tmp_eval_accuracy
                #eval_f1 += tmp_eval_f1
                #nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1

        self.model.train()
        eval_loss = eval_loss / nb_eval_steps
        #eval_accuracy = eval_accuracy / nb_eval_examples
        #eval_f1 = eval_f1 / nb_eval_examples

        print("  Validation Loss: {0:.4f}".format(eval_loss))
        #print("  Accuracy: {0:.4f}".format(eval_accuracy))
        #print("  F1 score: {0:.4f}".format(eval_f1))

        return eval_loss, eval_accuracy, eval_f1



    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version_full_bs_32_20_epochs_with_context/checkpoints/checkpoint_epoch_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int):
        best_loss = float("inf")
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                validation_loss, _ , _ = self._validate()
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self._save_checkpoint(epoch)
                    print(f"Epoch {epoch} | Best validation loss: {best_loss}")
            torch.distributed.barrier() # wait for all processes to finish the epoch  
            #free GPU memory
            torch.cuda.empty_cache()
                      


def load_data(rank,batch_size):
     #load the lists saved in deb_train_gpu_parallel.py
    # the lists saved full_inputs, inputs_ids, attention_masks and the targets in different files /home/daril_kw/data/input_ids.pkl, /home/daril_kw/data/attention_masks.pkl, /home/daril_kw/data/targets.pkl


    with open(DIR_INPUTS_IDS, 'rb') as f:
        input_ids = pickle.load(f)

    with open(DIR_ATTENTION_MASKS, 'rb') as f:
        attention_masks = pickle.load(f)

    with open(DIR_TARGETS, 'rb') as f:
        targets = pickle.load(f)


    targets_dict={}
    for i in range(len(targets)):
        if targets[i] not in targets_dict:
            targets_dict[targets[i]]=len(targets_dict)

    targets_input=[targets_dict[targets[i]] for i in range(len(targets))]

    train_data, _, train_targets, _ = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2)
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_targets,random_state=2023, test_size=0.1)

    train_masks, _, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(train_masks, train_targets,random_state=2023, test_size=0.1)


    print("Data conversion to tensors...\n")
    #on convertit les données en tenseurs
    train_inputs = torch.tensor(train_inputs).to(rank)
    train_labels = torch.tensor(train_labels).to(rank)
    train_masks = torch.tensor(train_masks).to(rank)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = prepare_dataloader(train_data, batch_size=batch_size)


    validation_inputs = torch.tensor(validation_inputs).to(rank)
    validation_labels = torch.tensor(validation_labels).to(rank)
    validation_masks = torch.tensor(validation_masks).to(rank)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data) # we don't use the DistributedSampler here because the validation is on a CPU
    validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)



    return train_dataloader, validation_dataloader

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size) 
    # we load the data
    train_dataloader, validation_dataloader = load_data(rank,batch_size)
    # we load the model
    model, optimizer = load_bert_model_and_tokenizer()
    # computing of the total number of steps
    total_steps = len(train_dataloader) * total_epochs
    # we load the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)
    # we create the trainer
    trainer = Trainer(model, train_dataloader, validation_dataloader, optimizer, rank, save_every, scheduler)
    # we train the model
    trainer.train(total_epochs)
    # we destroy the process group
    destroy_process_group()

    #save the model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version_full_bs_32_20_epochs_with_context')



if __name__ == "__main__":
    import json
    # import the config file
    with open("config_1_2_batch_size_32_20_epochs.json") as json_file:
        config = json.load(json_file)

    batch_size = config["batch_size"]
    epochs = config["num_epochs"]
    save_every = config["save_every"]
    

    world_size = torch.cuda.device_count()
    
    mp.spawn(main, args=(world_size, save_every, epochs, batch_size), nprocs=world_size, join=True)
    """
    children = []
    for i in range(world_size):
        subproc = mp.Process(target=main, args=(i, world_size, save_every, epochs, batch_size))
        children.append(subproc)
        subproc.start()

    for i in range(world_size):
        children[i].join()
    """
    
    
        

  



















































































# from transformers import  BertForSequenceClassification
# import json
# import os
# from sklearn.model_selection import train_test_split
# import torch
# import pickle
# import numpy as np
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
# from transformers import get_linear_schedule_with_warmup
# from torch.optim import AdamW
# from torch.utils.data.distributed import DistributedSampler


# from torch.distributed import init_process_group, destroy_process_group
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP



# DIR_INPUTS_IDS = '/home/daril/trajcbert/savings_for_parrallel_1_2/input_ids_f_833383.pkl'
# DIR_ATTENTION_MASKS = '/home/daril/trajcbert/savings_for_parrallel_1_2/attention_masks_833383_opti.pkl'
# DIR_TARGETS = '/home/daril/trajcbert/savings_for_parrallel_1_2/targets_833383_opti.pkl'
# PRETRAINED_MODEL_NAME = '/home/daril/trajcbert/savings_for_parrallel_1_2/model_before_training_opti_833383'

 
# def ddp_setup(rank, world_size):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes: it is the number of available GPUs
#     """
#     os.environ["MASTER_ADDR"] = "localhost" #ip address of master node
#     os.environ["MASTER_PORT"] = "12355"      #port of master node, this port is used by master node to listen to workers (can be any free port)
#     init_process_group(backend="nccl", rank=rank, world_size=world_size) # initialize the process group
#     torch.cuda.set_device(rank) # define the current GPU as the one to use

# def prepare_dataloader(dataset: Dataset,batch_size: int):
#     return DataLoader(dataset, 
#                       batch_size=batch_size, 
#                       shuffle=False, # Must be False when using DistributedSampler beacause it is already shuffled
#                     #   pin_memory=True # Automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.
#                       sampler = DistributedSampler(dataset) # Select a subset of the dataset (only works if shuffle=False)
      
      
#                     )


# def load_bert_model_and_tokenizer():
#     """
#     Load the pretrained BERT model and the tokenizer
#     """
#     #load the tokenizer
#     # tokenizer = BertTokenizer.from_pretrained('/home/daril/trajcbert/BERT_MODEL/tokenizer_augmented_full')

#     #load the model
#     model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)

#     optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8)
#     return model, optimizer

# class Trainer:


#     """
#     model: the model to train like BertForSequenceClassification
#     train_data: the training data
#     optimizer: the optimizer to use like AdamW
#     gpu_id: the id of the gpu to use
#     save_every: the number of epochs between each checkpoint
#     scheduler: the scheduler to use like get_linear_schedule_with_warmup

#     return: None

#     """
#     def __init__( 
#         self,
#         model: torch.nn.Module, 
#         train_data: DataLoader,
#         validation_data: DataLoader,
#         optimizer: torch.optim.Optimizer,
#         gpu_id: int, # the id of the gpu to use
#         save_every: int, 
#         scheduler: torch.optim.lr_scheduler.LambdaLR
#     ) -> None:
#         self.gpu_id = gpu_id
#         self.model = model.to(gpu_id)
#         self.train_data = train_data
#         self.validation_data = validation_data
#         self.optimizer = optimizer
#         self.save_every = save_every
#         self.scheduler = scheduler
#         self.model = DDP(model, device_ids=[gpu_id], )



#     def _run_batch(self,  b_input_ids,b_input_mask,b_labels): 
#         #we set the gradients to zero
#         self.model.zero_grad()
#         #we make the forward pass
#         outputs = self.model(b_input_ids,token_type_ids=None,attention_mask=b_input_mask,labels=b_labels)
#         # the output is a tuple containing the loss and the logits (the output of the last layer)
#         #we get the loss
#         loss = outputs[0]
#         #we make the backward pass
#         loss.backward()
#         #we clip the gradient to avoid exploding gradient
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#         #we update the parameters
#         self.optimizer.step()
#         #we update the learning rate
#         self.scheduler.step()
#         #we accumulate the loss
#         return loss.item()
       

    
#     def _run_epoch(self, epoch):
#         # we recover the batch size
#         b_sz = len(next(iter(self.train_data))[0]) # batch size
#         print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
#         self.train_data.sampler.set_epoch(epoch)
#         total_loss = 0.0
#         for batch in self.train_data:
#             batch = tuple(t.to(self.gpu_id) for t in batch)
#             #unpack the batch
#             input_ids, attention_mask, labels = batch
#             # input_ids = input_ids.to(self.gpu_id)
#             # attention_mask = batch["attention_mask"].to(self.gpu_id)
#             # labels = batch["labels"].to(self.gpu_id)
#             loss = self._run_batch(input_ids,attention_mask,labels)
#             total_loss += loss
#         average_loss = total_loss / len(self.train_data)
#         print(f"[GPU{self.gpu_id}] Epoch {epoch} | Average loss: {average_loss}")
#         return average_loss
    
#     def _accuracy(self, logits : np.ndarray, labels: np.ndarray) -> float:
#         predicted = np.argmax(logits, axis=1).flatten()
#         labels = labels.cpu().numpy()  # Convert torch.Tensor to numpy array
#         correct = (predicted == labels)
#         print(f"correct type: {type(correct)}  predited type: {type(predicted)}  labels type: {type(labels)}")
#         correct = correct.sum()
#         total = labels.size
#         accuracy = correct / total
#         return accuracy
        
#     def _validate(self):
#         model = self.model.module
#         model.eval()
#         #self.model.eval()
#         eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
#         nb_eval_steps, nb_eval_examples = 0, 0

#         with torch.no_grad():
#             for batch in self.validation_data:
#                 batch = tuple(t.to(self.gpu_id) for t in batch)
#                 b_input_ids, b_input_mask, b_labels = batch
#                 b_input_ids = b_input_ids.to(self.gpu_id)
#                 b_input_mask = b_input_mask.to(self.gpu_id)
#                 b_labels = b_labels.to(self.gpu_id)

#                 outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#                 #outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#                 loss = outputs.loss
            
                

#                 eval_loss += loss.item()
    

#                 #logits = logits.detach().cpu().numpy() # logits is a tensor on the GPU, we need to move it to the CPU and then to the memory
              
#                 #label_ids = b_labels.to('cpu').numpy() # same for the labels
#                 #tmp_eval_accuracy = self._accuracy(logits, b_labels)
#                 #tmp_eval_f1 = f1_score(label_ids, np.argmax(logits, axis=1), average='macro')

#                 #eval_accuracy += tmp_eval_accuracy
#                 #eval_f1 += tmp_eval_f1
#                 #nb_eval_examples += b_input_ids.size(0)
#                 nb_eval_steps += 1

#         self.model.train()
#         eval_loss = eval_loss / nb_eval_steps
#         #eval_accuracy = eval_accuracy / nb_eval_examples
#         #eval_f1 = eval_f1 / nb_eval_examples

#         print("  Validation Loss: {0:.4f}".format(eval_loss))
#         #print("  Accuracy: {0:.4f}".format(eval_accuracy))
#         #print("  F1 score: {0:.4f}".format(eval_f1))

#         return eval_loss, eval_accuracy, eval_f1




#     def _save_checkpoint(self, epoch):
#         ckp = self.model.module.state_dict()
#         PATH = f"checkpoint_epoch_{epoch}.pt"
#         torch.save(ckp, PATH)
#         print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


#     def train(self, max_epochs: int):
#         best_loss = float("inf")
#         for epoch in range(max_epochs):
#             self._run_epoch(epoch)
#             if self.gpu_id == 0 and epoch % self.save_every == 0:
#                 validation_loss, _ , _ = self._validate()
#                 if validation_loss < best_loss:
#                     best_loss = validation_loss
#                     self._save_checkpoint(epoch)
#                     print(f"Epoch {epoch} | Best validation loss: {best_loss}")
#             torch.distributed.barrier() # wait for all processes to finish the epoch            


# def load_data(rank,batch_size):
#      #load the lists saved in deb_train_gpu_parallel.py
#     # the lists saved full_inputs, inputs_ids, attention_masks and the targets in different files /home/daril_kw/data/input_ids.pkl, /home/daril_kw/data/attention_masks.pkl, /home/daril_kw/data/targets.pkl
  

#     with open(DIR_INPUTS_IDS, 'rb') as f:
#         input_ids = pickle.load(f)

#     with open(DIR_ATTENTION_MASKS, 'rb') as f:
#         attention_masks = pickle.load(f)

#     with open(DIR_TARGETS, 'rb') as f:
#         targets = pickle.load(f)

#     print("Generate the targets dictionary...\n")
#     targets_dict={}
#     for i in range(len(targets)):
#         if targets[i] not in targets_dict:
#             targets_dict[targets[i]]=len(targets_dict)
#     print("Targets dictionary generated\n")

#     targets_input=[targets_dict[targets[i]] for i in range(len(targets))]

#     train_data, _, train_targets, _ = train_test_split(input_ids, targets_input,random_state=2023, test_size=0.2)
#     train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_data, train_targets,random_state=2023, test_size=0.1)

#     train_masks, _, _, _ = train_test_split(attention_masks, targets_input,random_state=2023, test_size=0.2)
#     train_masks, validation_masks, _, _ = train_test_split(train_masks, train_targets,random_state=2023, test_size=0.1)


#     print("Data conversion to tensors...\n")
#     #on convertit les données en tenseurs
#     train_inputs = torch.tensor(train_inputs).to(rank)
#     train_labels = torch.tensor(train_labels).to(rank)
#     train_masks = torch.tensor(train_masks).to(rank)
#     train_data = TensorDataset(train_inputs, train_masks, train_labels)
#     train_dataloader = prepare_dataloader(train_data, batch_size=batch_size)


#     validation_inputs = torch.tensor(validation_inputs).to(rank)
#     validation_labels = torch.tensor(validation_labels).to(rank)
#     validation_masks = torch.tensor(validation_masks).to(rank)
#     validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
#     validation_sampler = SequentialSampler(validation_data)
#     validation_dataloader = DataLoader(validation_data,sampler=validation_sampler, batch_size=batch_size)



#     return train_dataloader, validation_dataloader

# def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
#     ddp_setup(rank, world_size) 
#     # we load the data
#     train_dataloader, validation_dataloader = load_data(rank,batch_size)
#     # we load the model
#     model, optimizer = load_bert_model_and_tokenizer()
#     # computing of the total number of steps
#     total_steps = len(train_dataloader) * total_epochs
#     # we load the scheduler
#     scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = total_steps)
#     # we create the trainer
#     trainer = Trainer(model, train_dataloader, validation_dataloader, optimizer, rank, save_every, scheduler)
#     # we train the model
#     trainer.train(total_epochs)
#     # we destroy the process group
#     destroy_process_group()

#     #save the model
#     model_to_save = model.module if hasattr(model, 'module') else model
#     model_to_save.save_pretrained('/home/daril/scratch/data/trajcbert/models/model_saved_parallel_version')



# if __name__ == "__main__":
#     # import the config file
#     with open("config.json") as json_file:
#         config = json.load(json_file)

#     batch_size = config["batch_size"]
#     epochs = config["num_epochs"]
#     save_every = config["save_every"]
    

#     world_size = torch.cuda.device_count()

#     mp.spawn(main, args=(world_size, save_every, epochs, batch_size), nprocs=world_size, join=True)
    

     


