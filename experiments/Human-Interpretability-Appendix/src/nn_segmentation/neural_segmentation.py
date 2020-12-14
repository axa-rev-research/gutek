# neural_segmentation.py
# Use pretrained model by Omri Koshorek et al. for text segmentation.
# https://www.aclweb.org/anthology/N18-2075.pdf
# https://github.com/koomri/text-segmentation



import torch
import numpy as np
import re
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings
from torch.serialization import SourceChangeWarning
# SourceChangeWarnings can be ignored : https://discuss.pytorch.org/t/sourcechangewarning-source-code-of-class/12126
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

class NeuralSegmenter:
    '''
    Class for Neural Text Segmentation.
    '''
    def __init__(self, model_path, word2vec_path, preload=True, paragraph_segment=False):
        '''
        Initialise the Neural Segmenter

        Args :
            model_path (str) : Path to the pretrained model (torch)
            word2vec_path (str) : Path to the pretrained word2vec model (gensim)
            preload (bool) : Already load the models in memory
        '''
        self._model_path = model_path
        self._word2vec_path = word2vec_path
        self._paragraph_segment = paragraph_segment
        if preload:
            with open(self._model_path, 'rb') as f:
                self.model = torch.load(f, map_location=torch.device('cpu'))
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self._word2vec_path, binary=True)
        else:
            self.model = None
            self.word2vec = None
    
    def get_word_embedding(self, word):
        '''
        Get the word2vec embedding for a word
        '''
        if not self.word2vec:
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self._word2vec_path, binary=True)

        if word in self.word2vec:
            # word exits
            return self.word2vec[word].reshape(1, 300)
        else:
            return self.word2vec['UNK'].reshape(1, 300)

    def get_segmentation_probs(self, sentences):
        '''
        Get the segmentation probabilities.

        Args :
            sentences (list of list of string) : List of sentences, each sentence being a list of words (string)
        '''
        # get the embeddings
        embeddings = [torch.FloatTensor(np.concatenate([self.get_word_embedding(w) for w in s])) for s in sentences]
        # load the model if necessary
        if not self.model:
            with open(self._model_path, 'rb') as f:
                self.model = torch.load(f, map_location=torch.device('cpu'))
        # get chunk probabilites
        output = self.model([embeddings])
        output_prob = softmax(output.data.cpu().numpy())
        return output_prob

    def unload_models(self):
        '''
        Remove the models from memory. They will be reloaded on the last call.
        '''
        self.model = None
        self.word2vec = None

    def __call__(self, text, threshold = 0.4):
        if self._paragraph_segment and len(re.split(r'\n *\n', text))>1:
            sents = []
            for paragraph in re.split(r'\n *\n', text):
                sents.extend(sent_tokenize(paragraph))
                if sents:
                    sents[-1] += '\n\n'
        else:
            sents = custom_tknzr.tokenize(text)
        sents_tokenized = [word_tokenize(s) for s in sents]
        probs = self.get_segmentation_probs(sents_tokenized)
        output_seg = np.append(probs[:, 1] > threshold, [False])

        chunks = ['']

        for s, r in zip(sents, output_seg):
            chunks[-1] += s + ' '
            if r:
                chunks.append('')
        return chunks


# some hacking to keep the newlines even when using nltk sent_tokenizer (curtesy to HugoMailhot 
# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer)
import nltk.tokenize.punkt as pkt
class CustomLanguageVars(pkt.PunktLanguageVars):

    _period_context_fmt = r'''
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))'''

custom_tknzr = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())
