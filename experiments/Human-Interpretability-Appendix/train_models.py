# train_models.py
# train LDA, HTMM & classifier
import warnings

warnings.filterwarnings(action="ignore")

import argparse

import os

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

import pyhtmm
import spacy
import gensim
import pickle

from src import nlp_feature_extraction, dataloader


def train_classifier(train_data, save_path):
    """
    Train and save the classifier
    """
    print("Fitting the classifier...")
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_features=None,
                    max_df=0.8,
                    min_df=4,
                    norm="l1",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", RandomForestClassifier()),
        ]
    )
    pipe.fit(train.text, train.target)
    print("Done, saving...")
    dump(pipe, os.path.join(save_path, "classifier.joblib"))
    print("Done!")


def train_LDA(train_data, save_path):
    """
    Train and save the LDA model
    """
    print("Training TopicTiling...")
    full_extractor = nlp_feature_extraction.NLPFeatureExtraction(model=("Lda", 100))
    full_extractor.fit(train_data.text, diagnostics=False, save=save_path)
    print("Done!")


def train_htmm(train_data, save_path, n_topics=120):
    """
    Train and save the htmm model
    """
    # training
    print("Training htmm")
    nlp = spacy.load("en")
    STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

    docs = list()

    for doc in nlp.pipe(train_data.text, n_threads=5, batch_size=10):
        doc = [
            [t.lemma_.lower() for t in s if t.is_alpha and not t.is_stop]
            for s in doc.sents
        ]

        docs.append(doc)

    corpus = [[word for sent in doc for word in sent] for doc in docs]
    dictionary = gensim.corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    docs_idx = [
        [[i for i in dictionary.doc2idx(s) if i != -1] for s in d] for d in docs
    ]

    docs = []
    for doc in docs_idx:
        d = pyhtmm.document._Document()
        for sent in doc:
            s = pyhtmm.sentence._Sentence("noneyves")
            for word in sent:
                s.add_word(word)
            d.add_sentence(s)
        if d.num_sentences > 0:
            docs.append(d)
    htmm = pyhtmm.htmm.EM(
        docs,
        len(dictionary.keys()),
        topics=n_topics,
        num_workers=8,
        alpha=1.5,
        iters=500,
    )
    htmm.infer()
    # save
    pickle.dump(htmm, open(os.path.join(save_path, "htmm.p"), "wb"), protocol=4)
    dictionary.save(os.path.join(save_path, "dict_htmm.p"))


if __name__ == "__main__":
    # parsing the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="On which dataset to train, must be one of [ng_transport, ng_religion, wiki, reuters, sentiment]",
    )
    parser.add_argument(
        "--do_htmm", help="Wether or not to also train the htmm", action="store_true"
    )
    args = parser.parse_args()
    # loading the data
    if args.dataset == "ng_transport":
        train, test = dataloader.load_20_newsgroups(
            clean=True, clean_test=True, dataset="transport"
        )
    elif args.dataset == "ng_religion":
        train, test = dataloader.load_20_newsgroups(
            clean=True, clean_test=True, dataset="transport"
        )
    elif args.dataset == "wiki":
        train, test = dataloader.load_wiki_city()
    elif args.dataset == "reuters":
        train, test = dataloader.load_reuters_news()
    elif args.dataset == "sentiment":
        train, test = dataloader.load_sentiment_dataset()
    else:
        raise Exception("The specified dataset does not exist")
    # train the classifier
    train_classifier(train, "../models/{}".format(args.dataset))

    # train the LDA
    train_LDA(train, "../models/{}".format(args.dataset))

    if args.do_htmm:
        train_htmm(train, "../models/{}".format(args.dataset))

