import warnings
import pandas as pd
import os
import json
import argparse
import pickle
import transformers
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from tokenization_kobert import KoBertTokenizer
from load_data import *

warnings.filterwarnings(action='ignore') 

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc,}

def train(cfg):
   
    results_path = os.path.join('./results', cfg['train_id_name'])
    if not os.path.exists(results_path):
      os.mkdir(results_path)

    os.environ['WANDB_PROJECT'] = 'KLUE_PROJECT'
    os.environ['WANDB_LOG_MODEL'] = 'true'

    MODEL_NAME = cfg['model_name']
    
    if MODEL_NAME == 'monologg/kobert':
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with open('../input/data/train/ner_tags.pickle', 'rb') as f:
        ner_tokens = pickle.load(f)
    special_tokens_dict = {'additional_special_tokens': ner_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    train_dataset = load_data(cfg['train_data_path'])
    dev_dataset = load_data(cfg['valid_data_path'])
    
    train_label = train_dataset['label'].values
    dev_label = dev_dataset['label'].values
    
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
    
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
#     model_config.vocab_size += len(ner_tokens)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)
    
    training_configs = cfg['train_args']

    training_args = TrainingArguments(
                        output_dir=results_path,          
                        save_total_limit=training_configs['save_total_limit'],
                        save_steps=training_configs['save_steps'],
                        num_train_epochs=training_configs['num_train_epochs'],
                        learning_rate=training_configs['learning_rate'],
                        per_device_train_batch_size=training_configs['per_device_train_batch_size'],
                        per_device_eval_batch_size=training_configs['per_device_eval_batch_size'],
                        warmup_steps=training_configs['warmup_steps'],
                        weight_decay=training_configs['weight_decay'],
                        logging_dir=training_configs['logging_dir'],
                        logging_steps=training_configs['logging_steps'],
                        evaluation_strategy=training_configs['evaluation_strategy'],
                        load_best_model_at_end=True,
                        )

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=RE_train_dataset,
                    eval_dataset=RE_dev_dataset,
                    compute_metrics=compute_metrics,
                    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience'],),]
                    )
    
    transformers.integrations.WandbCallback()
    
    print('Start Training.')
    trainer.train()
    print('Fininshed Training.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    with open((os.path.join('./config', args.config + '.json'))) as j:
      json_config = json.load(j)

    train(json_config)

if __name__ == '__main__':
    main()
