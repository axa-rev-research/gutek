# dataloader.py
# Load datasets. Each function returns a tuple of pandas dataframes (train, test). Each dataframe has the attributes `text` for the text in string format and 
# `target` for the target.

from sklearn.datasets import fetch_20newsgroups
import os, re
import pandas as pd
import numpy as np

def load_20_newsgroups(clean = False, clean_test = False):
    '''
    Return 20 newsgroups dataset. The dataset is modified for binary classifiation atheism (1) vs christian (0).
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    remove = ('headers', 'footers', 'quotes') if clean else ()
    remove_test = ('headers', 'footers', 'quotes') if clean_test else ()
    subset = ['alt.atheism', 'soc.religion.christian']
    train = fetch_20newsgroups(subset='train', categories=subset, remove=remove)
    test = fetch_20newsgroups(subset='test', categories=subset, remove=remove_test)
    return pd.DataFrame(data = {'target':train['target'], 'text':train['data']}),\
        pd.DataFrame(data = {'target':test['target'], 'text':test['data']})

def load_spam(path='/Users/b054of/Documents/datasets'):
    '''
    Return spam dataset. Target: Spam = 1, Non-Spam = 0
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    msg_folder = os.path.join(path, 'lingspam_public/bare')
    # get the parts
    parts = [folder for folder in os.listdir(msg_folder) if not ('.' in folder)]
    # get the filenames
    spam_files = [os.path.join(msg_folder, folder, file) for folder in parts for file in os.listdir(os.path.join(msg_folder, folder))\
        if 'spm' in file and '.txt' in file]
    msg_files = [os.path.join(msg_folder, folder, file) for folder in parts for file in os.listdir(os.path.join(msg_folder, folder)) \
        if '.txt' in file and not 'spm' in file]
    # build the dataframe
    spam_df = pd.DataFrame(data={'source': spam_files, 'target':1})
    msg_df = pd.DataFrame(data={'source': msg_files, 'target':0}) 
    df = spam_df.append(msg_df)
    # load the texts
    def load_txt(file):
        with open(file) as f:
            txt = f.read()
        return txt
    df['text'] = df.apply(lambda row: load_txt(row['source']), axis=1)
    # keep the last 2 folds as test set
    train = df.loc[~(df.source.str.contains('part9') | df.source.str.contains('part10'))].reset_index(drop=True)
    test = df.loc[(df.source.str.contains('part9') | df.source.str.contains('part10'))].reset_index(drop=True)
    # return the dataset
    return train[['text', 'target']], test[['text', 'target']]

def load_reuters_news(path='/Users/b054of/Documents/datasets'):
    '''
    Return Reuters News dataset. Target: earn = 1, Non-earn = 0
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
    df['text'] = df.body.str.replace('\x03', '').str.replace('\n', ' ').str.replace(re.compile('\s+'), ' ').str.replace(re.compile('Reuter\s*$'), '')
    # train-test split
    df = df.sample(frac=1, random_state=2020).reset_index(drop=True)
    split = int(0.8*len(df))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train[['text', 'target']].reset_index(drop=True), test[['text', 'target']].reset_index(drop=True)

def load_yelp(path='/Users/b054of/Documents/datasets', frac = 0.1):
    '''
    Return Yelp-2 dataset. Target: positive = 1, Negative = 0
    Parameters (Optional):
        path : string : The root path containing the datasets
        frac : float : The fraction of the dataset to use
    Returns:
        (train, test) : tuple of (pandas.DataFrame, pandas.DataFrame) : The train and test dataframes
    '''
    # load the data
    train = pd.read_csv(os.path.join(path, 'yelp/train.csv'),header=None, names = ['target', 'text'])
    test = pd.read_csv(os.path.join(path, 'yelp/test.csv'),header=None, names = ['target', 'text'])
    # subsample
    train = train.sample(frac = frac, random_state=2020).reset_index(drop=True)
    test = test.sample(frac = frac, random_state=2020).reset_index(drop=True)
    # clean target and text
    train['target'] = train['target']-1
    test['target'] = test['target']-1
    train['text'] = train.text.str.decode('unicode_escape')
    test['text'] = test.text.str.decode('unicode_escape')
    return train, test
