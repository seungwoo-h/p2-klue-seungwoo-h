import pandas as pd
import numpy as np
import re
import pickle
from pororo import Pororo
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = [str(i) for i in range(9)]
    return df

def create_ner_tagged_dataset(df):
    # Get ner tags
    print('Start getting ner tags...')
    ner = Pororo(task='ner', 'ko')
    ne_1, ne_2 = list(), list()
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx, :]
        ner_results = {k: v for k, v in ner(row['1'])}
        try:
            ne_1.append(ner_results[row['2']])
        except:
            ne_1.append(None)
        try:
            ne_2.append(ner_results[row['5']])
        except:
            ne_2.append(None)
    df['9'] = ne_1
    df['10'] = ner_2

    # Apply in main sentences
    new_s_lst = []
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx, :]
        s = row['1']
        new_s = s.replace(row['2'], '[{}] {} [/{}]'.format(row['9'], row['2'], row['9']))
        new_s = new_s.replace(row['5'], '[{}] {} [/{}]'.format(row['10'], row['5'], row['10']))
        new_s_lst.append(new_s)
    df['11'] = new_s_lst
    del new_s_lst

    # For special tokens
    ner_tokens = [i for i in df['9'].unique()]
    ner_tokens.extend([i for i in df['10'].unique()])
    ner_tokens = list(set(ner_tokens))
    ner_tokens_processed = []
    for token in ner_tokens:
        a = f'[{token}]'
        b = f'[/{token}]'
        ner_tokens_processed.append(a)
        ner_tokens_processed.append(b)

    # Save
    with open('ner_tag_tokens.pickle', 'wb') as f:
        pickle.dump(ner_tokens_processed, f)
    print('Ner tokens saved.')

    df['1'] = df['11']
    return df.iloc[:, :9]

def create_translated_dataset(df, target_lang='en', use_lower=True):
    mt = Pororo(task='translation', lang='multi')
    tqdm.pandas()
    df['1'] = df['1'].progress_apply(lambda x: mt(x, src='ko', tgt=target_lang)) # main sentence
    df['2'] = df['2'].progress_apply(lambda x: mt(x, src='ko', tgt=target_lang)) # entity 1
    df['5'] = df['5'].progress_apply(lambda x: mt(x, src='ko', tgt=target_lang)) # entity 2
    if use_lower:
        df['1'] = df['1'].apply(lambda x: x.lower())
        df['2'] = df['2'].apply(lambda x: x.lower())
        df['5'] = df['5'].apply(lambda x: x.lower())
    return df

def main(task, path):
    df = load_data(path)
    if task == 'ner':
        pass
    elif task == 'entity':
        pass
    elif task == 'translate':
        pass

if __name__ == '__main__':
    pass