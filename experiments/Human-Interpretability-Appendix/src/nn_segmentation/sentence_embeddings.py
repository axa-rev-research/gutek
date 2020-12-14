# sentence_embeddings.py

# imports
import os
import time
import sys
import re
from subprocess import call
import numpy as np
from nltk import TweetTokenizer
from nltk.tokenize.stanford import StanfordTokenizer
from scipy.spatial.distance import cosine as cosine_distance

# define paths TODO: INIT
FASTTEXT_EXEC_PATH = os.path.abspath("/Users/b054of/Documents/packages/sent2vec-master/fasttext")

BASE_SNLP_PATH = "/Users/b054of/Documents/packages/stanford_nlp/pos/"
SNLP_TAGGER_JAR = os.path.join(BASE_SNLP_PATH, "stanford-postagger.jar")

MODEL_WIKI_UNIGRAMS = os.path.abspath("/Users/b054of/Documents/packages/sent2vec-master/wiki_unigrams.bin")
MODEL_WIKI_BIGRAMS = os.path.abspath("/Users/b054of/Documents/packages/sent2vec-master/wiki_bigrams.bin")
MODEL_TWITTER_UNIGRAMS = os.path.abspath('/Users/b054of/Documents/packages/sent2vec-master/twitter_unigrams.bin')
MODEL_TWITTER_BIGRAMS = os.path.abspath('/Users/b054of/Documents/packages/sent2vec-master/twitter_bigrams.bin')
class SentenceEmbedder:
    def __tokenize(self, tknzr, sentence, to_lower=True):
        """Arguments:
            - tknzr: a tokenizer implementing the NLTK tokenizer interface
            - sentence: a string to be tokenized
            - to_lower: lowercasing or not
        """
        sentence = sentence.strip()
        sentence = ' '.join([self.__format_token(x) for x in tknzr.tokenize(sentence)])
        if to_lower:
            sentence = sentence.lower()
        sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
        sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
        filter(lambda word: ' ' not in word, sentence)
        return sentence

    def __format_token(self, token):
        """"""
        if token == '-LRB-':
            token = '('
        elif token == '-RRB-':
            token = ')'
        elif token == '-RSB-':
            token = ']'
        elif token == '-LSB-':
            token = '['
        elif token == '-LCB-':
            token = '{'
        elif token == '-RCB-':
            token = '}'
        return token

    def __tokenize_sentences(self, tknzr, sentences, to_lower=True):
        """Arguments:
            - tknzr: a tokenizer implementing the NLTK tokenizer interface
            - sentences: a list of sentences
            - to_lower: lowercasing or not
        """
        return [self.__tokenize(tknzr, s, to_lower) for s in sentences]

    def __get_embeddings_for_preprocessed_sentences(self, sentences, model_path, fasttext_exec_path):
        """Arguments:
            - sentences: a list of preprocessed sentences
            - model_path: a path to the sent2vec .bin model
            - fasttext_exec_path: a path to the fasttext executable
        """
        timestamp = str(time.time())
        test_path = os.path.abspath('./'+timestamp+'_fasttext.test.txt')
        embeddings_path = os.path.abspath('./'+timestamp+'_fasttext.embeddings.txt')
        self.__dump_text_to_disk(test_path, sentences)
        call(fasttext_exec_path+
            ' print-sentence-vectors '+
            model_path + ' < '+
            test_path + ' > ' +
            embeddings_path, shell=True)
        embeddings = self.__read_embeddings(embeddings_path)
        os.remove(test_path)
        os.remove(embeddings_path)
        assert(len(sentences) == len(embeddings))
        return np.array(embeddings)

    def __read_embeddings(self, embeddings_path):
        """Arguments:
            - embeddings_path: path to the embeddings
        """
        with open(embeddings_path, 'r') as in_stream:
            embeddings = []
            for line in in_stream:
                line = '['+line.replace(' ',',')+']'
                embeddings.append(eval(line))
            return embeddings
        return []

    def __dump_text_to_disk(self, file_path, X, Y=None):
        """Arguments:
            - file_path: where to dump the data
            - X: list of sentences to dump
            - Y: labels, if any
        """
        with open(file_path, 'w') as out_stream:
            if Y is not None:
                for x, y in zip(X, Y):
                    out_stream.write('__label__'+str(y)+' '+x+' \n')
            else:
                for x in X:
                    out_stream.write(x+' \n')

    def get_sentence_embeddings(self, sentences, ngram='bigrams', model='concat_wiki_twitter'):
        """ Returns a numpy matrix of embeddings for one of the published models. It
        handles tokenization and can be given raw sentences.
        Arguments:
            - ngram: 'unigrams' or 'bigrams'
            - model: 'wiki', 'twitter', or 'concat_wiki_twitter'
            - sentences: a list of raw sentences ['Once upon a time', 'This is another sentence.', ...]
        """
        wiki_embeddings = None
        twitter_embbedings = None
        tokenized_sentences_NLTK_tweets = None
        tokenized_sentences_SNLP = None
        if model == "wiki" or model == 'concat_wiki_twitter':
            tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
            s = ' <delimiter> '.join(sentences) #just a trick to make things faster
            tokenized_sentences_SNLP = self.__tokenize_sentences(tknzr, [s])
            tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
            assert(len(tokenized_sentences_SNLP) == len(sentences))
            if ngram == 'unigrams':
                wiki_embeddings = self.__get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                        MODEL_WIKI_UNIGRAMS, FASTTEXT_EXEC_PATH)
            else:
                wiki_embeddings = self.__get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                        MODEL_WIKI_BIGRAMS, FASTTEXT_EXEC_PATH)
        if model == "twitter" or model == 'concat_wiki_twitter':
            tknzr = TweetTokenizer()
            tokenized_sentences_NLTK_tweets = self.__tokenize_sentences(tknzr, sentences)
            if ngram == 'unigrams':
                twitter_embbedings = self.__get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                        MODEL_TWITTER_UNIGRAMS, FASTTEXT_EXEC_PATH)
            else:
                twitter_embbedings = self.__get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                        MODEL_TWITTER_BIGRAMS, FASTTEXT_EXEC_PATH)
        if model == "twitter":
            return twitter_embbedings
        elif model == "wiki":
            return wiki_embeddings
        elif model == "concat_wiki_twitter":
            return np.concatenate((wiki_embeddings, twitter_embbedings), axis=1)
        sys.exit(-1)