

from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np
import h3
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef
import os
import json


#directories :
#-------------

#To get the model directory, we take the last checkpoint of the training saved. For that we use the config file to get the number of epochs maximum and we check if the checkpoint exists
with open("/home/daril_kw/trajcbert/trajcbert/config_test_gene.json") as json_file:
        config = json.loads(json_file.read())
epoch = config['num_epochs']

#we check if '/home/daril_kw/data/test/temp_file/checkpoint_epoch_{epoch}.pt' exists, otherwise we take the last checkpoint before
while not os.path.exists('/home/daril_kw/data/test/temp_file/checkpoint_epoch_{epoch}.pt') and epoch > 0:
  epoch -= 1
   
DIR_MODEL_NOT_TRAINED = '/home/daril_kw/data/model_resized_embeddings_test'
TRAINED_MODEL_PATH = f"/home/daril_kw/data/test/temp_file/checkpoint_epoch_1.pt"
DIR_TEST_DATALOADER = '/home/daril_kw/data/test_dataloader_parallel_gene.pt'



def flat_accuracy(preds, labels):
    """this function computes the accuracy of the predictions"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_f1(preds, labels):
    """this function computes the f1 score of the predictions"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat,pred_flat,average='macro')

 
def flat_matthews(preds, labels):
    """this function computes the matthews correlation coefficient of the predictions"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return matthews_corrcoef(labels_flat,pred_flat)

def test_autoregressively(prediction_dataloader, model, min_traj_rate, target_dict, device):
  """this function will predict the labels of the test set autoregressively"""

  model.to(device)
  model.eval()

  all_predictions = [[] for i in range(len(prediction_dataloader.dataset))]
  all_predictions_detokenized = [[] for i in range(len(prediction_dataloader.dataset))]
  all_true_labels = [[] for i in range(len(prediction_dataloader.dataset))]
  all_true_labels_coord = []
  
  with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(prediction_dataloader)):
      batch = batch[0].to(device) 
      #we do a loop on the trajectories of the batch
      print(f"there are {len(batch)} trajectories in the batch")
      for line in range(len(batch)):
      #we are going to predict the trajectory named traj autoregressively from the point nb_tokens to the end of the trajectory
        #traj is the trajectory we are going to predict ie contains the real trajectory minus the context tokens
        #line is the whole input of the model ie contains the cls token + context tokens + the real trajectory + the SEP token + the pad tokens
        #we remove the first token (CLS)
        traj = batch[line][1:]
        #we remove the pad tokens (0s)
        traj = traj[traj!=0]
        #we remove the last token (SEP)
        traj = traj[:-1]

        #we identify the first token of the trajectory (the first token that is not a context token) knowing that there are six context tokens
        first_token_traj = 6

        #we get the length of the trajectory (ie the number of tokens of the trajectory from the first token of the trajectory to the end of the trajectory)
        nb_tokens_traj = len(traj)-first_token_traj
        print(f"there are {nb_tokens_traj} tokens in the trajectory")
        first_token_to_predict = first_token_traj+int(min_traj_rate*nb_tokens_traj)
        print("the length of the trajectory is ", nb_tokens_traj)
        print(f"we are going to predict the tokens from the point {first_token_to_predict} to the end of the trajectory, ie we predict {nb_tokens_traj-first_token_to_predict} tokens")
        
        #we take this token as the target token, remove every token after this token included and pad the input so that it has the same length as the input of the model ie 512
        #traj_i must contain the input of the model ie the cls token + context tokens + the real trajectory + the SEP token
        traj_i = traj[:first_token_to_predict]
        traj_i_padded = torch.nn.functional.pad(traj_i, (0,512-len(traj_i)), 'constant', 0)

        #we get the list of the true tokens of the part of the trajectory we are going to predict
        #ie the trajectory from the point 
        list_true_tokens = traj[first_token_to_predict:]
        print(f"the list of true tokens is {list_true_tokens}")
        print(f"the length of the list of true tokens is {len(list_true_tokens)}")
        #we get the nimbers instead of the tensors
        list_true_tokens = list_true_tokens.tolist()
        #we get the ids of the true tokens
        list_true_tokens_ids = [target_dict[token] for token in list_true_tokens]
        
      
        all_true_labels[batch_idx].append(list_true_tokens)
        #we get the list of the detokenized true tokens of the trajectory, ie we get the coordinates of the tokens if it is not the sep token
        #list_true_tokens_detokenized = [h3.h3_to_geo(list(target_dict.keys())[list(target_dict.values()).index(token)]) if list(target_dict.keys())[list(target_dict.values()).index(token)] != '[SEP]' else None for token in list_true_tokens]
        
        #the below loop is going to predict the tokens of the trajectory from the point first_token_traj to the end of the trajectory
        for index_token_to_predict in range(first_token_to_predict,nb_tokens_traj):
          print(f"we predict the token number {index_token_to_predict} on {nb_tokens_traj}")
        
          #we get the attention mask associated to the input that is going to be 1s for the tokens that are not padded and 0s for the tokens that are padded
          att_mask = torch.cat((torch.ones(len(traj_i)), torch.zeros(512-len(traj_i)))).to(device)
        
          #we unsqueeze the input and the attention mask to get the right shape for the model ie (1,512)
          att_mask = att_mask.unsqueeze(0)
          traj_i_padded = traj_i_padded.unsqueeze(0)
          print(f"the length of the list of true tokens ids is {len(list_true_tokens_ids)}"odel
          outputs = model(input_ids=traj_i_padded, token_type_ids=None, attention_mask=att_mask, labels=target) # we need the -first_token_traj because the labels are the true tokens of the trajectory from the point first_token_traj to the end of the trajectory
          #we get the logits
          logits = outputs[1].detach().cpu().numpy()
          #we get the predicted token traj_i_padded = torch.cat((traj_i_padded, torch.tensor([predicted_token])))
          predicted_token = np.argmax(logits) #the preicted token is the id in the targets_dict
          #we add the predicted token to the list of predictions
          all_predictions[batch_idx].append(predicted_token)
          #we get the detokenized predicted token ie from the id of the token we get the token and then we get the coordinates of the token if it is not the sep token
          #predicted_token_detokenized = h3.h3_to_geo(list(target_dict.keys())[list(target_dict.values()).index(predicted_token)]) if list(target_dict.keys())[list(target_dict.values()).index(predicted_token)] != '[SEP]' else None
          #we add the detokenized predicted token to the list of detokenized predictions
          #all_predictions_detokenized[batch_idx].append(predicted_token_detokenized)
          #we add the predicted to the input by taking the input of the model and adding the predicted token at the end
          #we get the traj from traj_i_padded
          

          traj_i_padded=traj[:first_token_to_predict]
          #we get all the predicted tokens for this line
          for i in range(len(all_predictions[batch_idx])):
            #print(f"we add the token number {i} to the input")
            traj_i_padded = torch.cat((traj_i_padded, torch.tensor([all_predictions[batch_idx][i]])))
          
          traj_i_padded = torch.nn.functional.pad(traj_i_padded, (0,512-len(traj_i_padded)), 'constant', 0)

         
  #return all_predictions, all_predictions_detokenized, all_true_labels, all_true_labels_coord
  return all_predictions, all_predictions_detokenized, all_true_labels



def calculate_MAD_score(all_predictions_detokenized, all_true_labels_coord):
  """this function calculates the MAD score on the coordinates of the predictions"""
  scores_coord = []
  for i in range(len(all_predictions_detokenized)):
    scores_coord.append(np.median([np.abs(all_predictions_detokenized[i][j][0]-all_true_labels_coord[i][j][0]) + np.abs(all_predictions_detokenized[i][j][1]-all_true_labels_coord[i][j][1]) for j in range(len(all_predictions_detokenized[i]))]))
  MAD_score = np.mean(scores_coord)
  return MAD_score

   
   
def calculate_MSE_score(all_predictions_detokenized, all_true_labels_coord):
  """this function calculates the MSE score on the coordinates of the predictions"""
  scores_coord_MSE = []
  for i in range(len(all_predictions_detokenized)):
    scores_coord_MSE.append(np.mean([(all_predictions_detokenized[i][j][0]-all_true_labels_coord[i][j][0])**2 + (all_predictions_detokenized[i][j][1]-all_true_labels_coord[i][j][1])**2 for j in range(len(all_predictions_detokenized[i]))]))
  MSE_score = np.mean(scores_coord_MSE)
  return MSE_score

#save flat_list_inputs_test


def main():
  device = torch.device("cpu") 

  with open("/home/daril_kw/trajcbert/trajcbert/parameter_sep_dup.json") as json_file:
      config_param = json.loads(json_file.read())

  min_traj_rate = config_param["rate_min_traj_input"]


  #load targets_input and targets_dict
  #the targets dict works like this : the key is the token and the value is the id of the token
    #if we want to get the id of a token we just have to do targets_dict[token]
    #if we want to get the token of an id we just have to do list(targets_dict.keys())[list(targets_dict.values()).index(id)]
  DIR_TARGET_DICT ='/home/daril_kw/data/targets_dict_whole.json'
  targets_dict = torch.load(DIR_TARGET_DICT)


  #load the prediction_dataloader
  #prediction_dataloader = torch.load('/home/daril_kw/data/pred_dataloader_v_small.pt')
  prediction_dataloader = torch.load(DIR_TEST_DATALOADER, map_location=device)

  # we load the model
  model = BertForSequenceClassification.from_pretrained(DIR_MODEL_NOT_TRAINED)
  state_dict = torch.load(TRAINED_MODEL_PATH, map_location=device)
  model.load_state_dict(state_dict)
  print("we evaluate")
  model.eval()

  #we predict
  all_predictions, all_predictions_detokenized, all_true_labels = test_autoregressively(prediction_dataloader, model, min_traj_rate, targets_dict, device)

  #we get the list of the coordinates associated to all_true_labels
  all_true_labels_coord = []
  #reminder : targets_dict[token] = id of the token and list(targets_dict.keys())[list(targets_dict.values()).index(id)] = token
  #for i in range(len(all_true_labels)):
  #  all_true_labels_coord.append([h3.h3_to_geo(targets_dict[all_true_labels[i][j]]) for j in range(len(all_true_labels[i]))])
  
  #we get the accuracy of the predictions
  #we manage the format of the predictions and the true labels
  all_predictions = [item for sublist in all_predictions for item in sublist]
  all_true_labels = [item for sublist in all_true_labels for item in sublist]

  accuracy = flat_accuracy(np.array(all_predictions), np.array(all_true_labels))
  #we want the accumulated loss witch is the loss of the predictions calculated with the cross entropy criterion 
  #we get the accumulated loss
  accumulated_loss = np.sum(np.array(all_predictions) == np.array(all_true_labels))/len(np.array(all_predictions))

  print("accuracy : ", accuracy)
  print("accumulated loss : ", accumulated_loss)

  #we can compare the coordinates of the predictions with the coordinates of the true labels
  #we use the MAD metric : we get the median of the absolute difference between the coordinates of the predictions and the coordinates of the true labels
  #MAD_score = calculate_MAD_score(all_predictions_detokenized, all_true_labels_coord)

  #we get the scores on the coordinates using the MSE criterion
  #MSE_score = calculate_MSE_score(all_predictions_detokenized, all_true_labels_coord)

  


  print("MAD score : ", MAD_score)
  print("MSE score : ", MSE_score)
  #the difference between the MSE and the MAD is that the MSE is more sensitive to outliers

     
      

if __name__ == "__main__":
    main()