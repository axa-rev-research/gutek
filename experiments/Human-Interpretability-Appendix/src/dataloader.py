# dataloader.py
# Load datasets. Each function returns a tuple of pandas dataframes (train, test). Each dataframe has the attributes `text` for the text in string format and 
# `target` for the target.

from sklearn.datasets import fetch_20newsgroups
import os, re
import pandas as pd
import numpy as np
import json
import warnings

def load_20_newsgroups(clean = False, clean_test = False, dataset = 'religion'):
    '''
    Return 20 newsgroups dataset. The dataset is modified for binary classifiation atheism (1) vs christian (0).
    Parameters:
        clean (bool) : if cleaning should be performed in the training set
        clean_test (bool) : if cleaning should be performed in the test set
        dataset (string) : which dataset to use : 
            'religion' : predict christian (1) vs atheist (0)
            'transport' : predict motorcycles (1) vs autos (0)
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    remove = ('headers', 'footers', 'quotes') if clean else ()
    remove_test = ('headers', 'footers', 'quotes') if clean_test else ()
    subset = ['alt.atheism', 'soc.religion.christian'] if dataset=='religion' else ['rec.motorcycles', 'rec.autos']
    train = fetch_20newsgroups(subset='train', categories=subset, remove=remove)
    test = fetch_20newsgroups(subset='test', categories=subset, remove=remove_test)
    print('Train target names : {}'.format(train['target_names']))
    print('Test target names : {}'.format(test['target_names']))
    return pd.DataFrame(data = {'target':train['target'], 'text':train['data']}),\
        pd.DataFrame(data = {'target':test['target'], 'text':test['data']})

def load_reuters_news(path='../data'):
    '''
    Return Reuters News dataset. Target: earn = 1, Non-earn = 0

    Parameters (Optional):
        path : string : The root path containing the datasets
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    # load the dataframe
    def load_df(path):
        df = pd.read_json(path)[['topics', 'body']].dropna().reset_index(drop=True)
        df['target'] = df.apply(lambda row: int('earn' in row['topics']), axis=1)
        return df
    d = pd.DataFrame()
    for i, f in enumerate(os.listdir(os.path.join(path, 'reuters_json/'))):
        if not '.json' in f:
            continue
        d = d.append(load_df(os.path.join(path, 'reuters_json', f)))
    d.reset_index(inplace=True)
    df = d
    # perform some data-cleaning
    #df['text'] = df.body.str.replace('\x03', '').str.replace('\n', ' ').str.replace(re.compile('\s+'), ' ').str.replace(re.compile('Reuter\s*$'), '')
    df['text'] = df.body.str.replace('\x03', '').str.replace(re.compile('\s+'), ' ').str.replace(re.compile('Reuter\s*$'), '')
    # train-test split
    df = df.sample(frac=1, random_state=2020).reset_index(drop=True)
    split = int(0.8*len(df))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train[['text', 'target']].reset_index(drop=True), test[['text', 'target']].reset_index(drop=True)

def load_sentiment_dataset(path='../data'):
    '''
    Return sentiment dataset. Target: positive = 1, Negative = 0
    Parameters (Optional):
        path : string : The root path containing the datasets
    Returns:
        df : pandas.DataFrame : The dataset, containing fold information in 'fold' (10 folds) of 200 balanced samples each
    '''
    # list the files to load
    neg = os.listdir(os.path.join(path, 'sentiment/review_polarity/txt_sentoken', 'neg'))
    pos = os.listdir(os.path.join(path, 'sentiment/review_polarity/txt_sentoken', 'pos'))

    # function for loading a data sample
    def build_sample(file, folder, polarity):
        fold = int(file[2])
        with open(os.path.join(folder,file)) as f:
            text = f.read()
        return (fold, text, polarity)

    # load the data
    data_pos = [build_sample(file, os.path.join(path, 'sentiment/review_polarity/txt_sentoken', 'pos'), 1) for file in pos]
    data_neg = [build_sample(file, os.path.join(path, 'sentiment/review_polarity/txt_sentoken', 'neg'), 0) for file in neg]

    # build the dataframe
    df = pd.DataFrame(data_pos, columns=['fold', 'text', 'target'])\
                .append(pd.DataFrame(data_neg, columns=['fold', 'text', 'target'])).reset_index(drop=True)

    return df.loc[df.fold<8].reset_index(drop=True), df.loc[df.fold>=8].reset_index(drop=True)
 
def load_wiki_city(path='../data'):
    '''
    Return Wikipeida City region classification dataset. Target: North America = 1, Rest of the World = 0

    Parameters (Optional):
        path : string : The root path containing the datasets
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    df = pd.read_csv(os.path.join(path, 'wikipedia/wiki_cities_v2.csv'))
    df = df.sample(frac=1., random_state=2020).dropna().reset_index(drop=True)
    split = int(0.8*len(df))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test
