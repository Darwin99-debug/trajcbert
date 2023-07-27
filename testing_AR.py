

from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification
import torch
import numpy as np
import h3
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef

device = torch.device("cpu")

model_dir = '/home/daril_kw/data/model_trained_cpu_version'
data_dir = '/home/daril_kw/data/data_format_v_small.csv'


    
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

def test_autoregressively(prediction_dataloader, targets_input,target_dict, model):
    """this function will predict the labels of the test set autoregressively"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = [[] for i in range(len(prediction_dataloader.dataset))]
    all_predictions_detokenized = [[] for i in range(len(prediction_dataloader.dataset))]
    all_true_labels = [[] for i in range(len(prediction_dataloader.dataset))]
    
    with torch.no_grad():
        for batch in tqdm(prediction_dataloader):
            batch = batch[0].to(device)
            #we get 50% of the length of the trajectory
            nb_tokens = int(batch.shape[1]/2)
            #we are going to predict the second half of the trajectory autoregressively
            for i in range(nb_tokens,len(batch[0])):
                batch_i = batch[:,:nb_tokens+i]
                #we add the last token predicted to the input
                if i != 0:
                  batch = torch.cat((batch, torch.tensor([all_predictions[i-1]])), dim=1)
                #we padd the input so that it has the same length as the input of the model ie 512
                batch = torch.nn.functional.pad(batch, (0,512-nb_tokens), 'constant', 0)
                #we create the attention mask associated to the input that is going to be 1s for the tokens that are not padded and 0s for the tokens that are padded
                #for that we concatenate a tensor of 1s of size nb_tokens and a tensor of 0s of size 512-nb_tokens
                attention_mask = torch.cat((torch.ones(nb_tokens), torch.zeros(512-nb_tokens))).to(device)
                #we get the target token for this iteration 
                target_token = batch[:,nb_tokens+i]
                #we predict the target token
                outputs = model(batch, token_type_ids=None, attention_mask=attention_mask, labels=target_token)
                #we get the logits
                logits = outputs[1].detach().cpu().numpy()
                #we get the predicted token
                predicted_token = np.argmax(logits)
                #we add the predicted token to the list of predictions
                all_predictions[i].append(predicted_token)
                #we get the detokenized predicted token
                predicted_token_detokenized = h3.h3_to_geo(target_dict[predicted_token])
                #we add the detokenized predicted token to the list of detokenized predictions
                all_predictions_detokenized[i].append(predicted_token_detokenized)
                #we get the true token
                true_token = target_token.detach().cpu().numpy()
                #we add the true token to the list of true tokens
                all_true_labels[i].append(true_token)
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
  #load targets_input and targets_dict
  targets_input = torch.load('/home/daril_kw/data/AR/targets_input_v_small_AR.pt')
  targets_dict = torch.load('/home/daril_kw/data/AR/targets_dict_v_small_AR.pt')

  #load the prediction_dataloader
  prediction_dataloader = torch.load('/home/daril_kw/data/pred_dataloader_v_small.pt')

  # we load the model
  model = BertForSequenceClassification.from_pretrained(model_dir)
  print("we evaluate")
  model.eval()

  #we predict
  all_predictions, all_predictions_detokenized, all_true_labels = test_autoregressively(prediction_dataloader, targets_input, targets_dict, model)

  #we get the list of the coordinates associated to all_true_labels
  all_true_labels_coord = []
  for i in range(len(all_true_labels)):
    all_true_labels_coord.append([h3.h3_to_geo(targets_dict[all_true_labels[i][j]]) for j in range(len(all_true_labels[i]))])
  
  #we can compare the coordinates of the predictions with the coordinates of the true labels
  #we use the MAD metric : we get the median of the absolute difference between the coordinates of the predictions and the coordinates of the true labels
  MAD_score = calculate_MAD_score(all_predictions_detokenized, all_true_labels_coord)

  #we get the scores on the coordinates using the MSE criterion
  MSE_score = calculate_MSE_score(all_predictions_detokenized, all_true_labels_coord)

  #the difference between the MSE and the MAD is that the MSE is more sensitive to outliers

     
      

