from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os
import json
import wandb

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    print(data)
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          # token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):    
    with open((os.path.join('./config', args.config + '.json'))) as j:
        cfg = json.load(j)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    TOK_NAME = cfg['model_name']
    tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
    with open('../input/data/train/ner_tags.pickle', 'rb') as f:
        ner_tokens = pickle.load(f)
    special_tokens_dict = {'additional_special_tokens': ner_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # load my model
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name'])
    model.resize_token_embeddings(len(tokenizer))
    
    wandb.login()
    run = wandb.init(project="KLUE_PROJECT")
    my_model_artifact = run.use_artifact(args.wandb_version)
    model_dir = my_model_artifact.download()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.parameters
    model.to(device)

    # load test datset
    # test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset_dir = args.path
    test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    # predict answer
    pred_answer = inference(model, test_dataset, device)

    # make csv file with predicted answer
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv('./prediction/submission_{}.csv'.format(cfg['train_id_name']), index=False)
    print('Done.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  parser.add_argument('--wandb_version', type=str)
  parser.add_argument('--path', type=str)
  args = parser.parse_args()
  main(args)
  
