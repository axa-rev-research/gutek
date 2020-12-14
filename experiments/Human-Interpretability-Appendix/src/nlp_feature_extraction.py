# nlp_feature_extraction.py
# extract topic distribution

import gensim
import nltk
import pandas as pd
import numpy as np
import logging
import warnings
import os
from . import nlp_pipe as pipe

class NLPFeatureExtraction:
    def __init__(self, smoothing = None, model= ('Lda', 50), additional_features = [], load=None, **kwargs):
        '''
        Initialise Feature Extractor
        Parameters:
        Optional:
            smoothing : float in (0,1): parameter controlling the smoothing
            model : tuple of (string, int) : 'Lda' or 'HDP', the nlp clustering model to use, the integer gives the number of clusters to use
            additional_features : list of function handles : additional feature extraction to be applied to each topic
        '''
        # setup class
        self.__nlp_pipe = pipe.NLPPipe(**kwargs)
        self.__trained = False
        self.__smoothing = smoothing
        self.__model = model
        self.__n_states = model[1]
        self.__additional_features = additional_features

        # initialize logging
        if os.path.isfile('gensim.log'):
            os.remove('gensim.log')
        open('gensim.log','x').close()
        logging.basicConfig(filename='gensim.log',
                            format='%(asctime)s : %(levelname)s : %(message)s', 
                            level=logging.INFO)
        if load:
            self.__topicmodel = gensim.models.LdaModel.load(os.path.join(load, 'lda'))
            self.__nlp_pipe.load(load)
            self.__trained = True

    def fit(self, texts, diagnostics=False, save=None):
        '''
        Train the model.
        Parameters:
            texts : list of strings : the texts to use. Should be oneline, ie not containing newline '\\n'
        Optional:
            diagnostics : whether diagnostics shold be done for the topic model.
        '''
        # fit the nlp pipe
        train = self.__nlp_pipe.fit_transform(texts)

        # fit the topic model
        self.__fit_lda(train)
        self.__trained=True

        if save:
            self.__topicmodel.save(os.path.join(save, 'lda'))
            self.__nlp_pipe.save(save)

    def __fit_lda(self, train):
        # Set training parameters.
        chunksize = 2000
        passes = 100
        iterations = 400

        # Make a index to word dictionary.
        temp = self.__nlp_pipe.dictionary[0]  # This is only to "load" the dictionary.
        id2word = self.__nlp_pipe.dictionary.id2token

        self.__topicmodel = gensim.models.LdaModel(
            corpus=train,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=self.__model[1],
            passes=passes,
            eval_every=1,
            minimum_probability=0,
            random_state=1
        )

    def predict_class(self, text):
        '''
        Predict class probabilities for a piece of text
        '''
        if not self.__trained:
            raise RuntimeError('The transformer has not been trained yet')
        bow = self.__nlp_pipe.transform([text])
        return self.__topicmodel.get_document_topics(bow[0])#self.__topicmodel[bow[0]]

    
        




