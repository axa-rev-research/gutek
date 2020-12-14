# topictiling.py
# topic tiling for text segmentation
# https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2012-riedletal-acl-srw.pdf

import numpy as np
import spacy
from scipy.signal import argrelmin
from nltk import sent_tokenize
import nltk.tokenize.punkt as pkt
import matplotlib.pyplot as plt

from . import nlp_feature_extraction
import re

METHOD_KL = 0
METHOD_COSINE_SIMILARITY = 1 

class Coherence:
    def __init__(self, method=METHOD_KL):
        self.get_score = self._kl_symmetric if method == METHOD_KL else self._cos_sim

    def _kl(self,u,v):
        return np.sum(u*np.log(u/v))
    
    def _kl_symmetric(self,u,v):
        return -self._kl(u,v)-self._kl(v,u)

    def _cos_sim(self,u,v):
        return np.dot(u,v)

    def __call__(self, u, v):
        return self.get_score(u,v)


class TopicTiling:
    def __init__(self, coherence, feature_extractor):
        self._coherence = coherence
        self._feature_extractor = feature_extractor

    def _get_topic_dist(self, text):
        return np.array([x[1] for x in self._feature_extractor.predict_class(text)])

    def _tile(self, sentences, w=2, s=1, k=1, agg=min, diagnostics=False):
        c = []
        # find the coherence scores
        for i in range(1, len(sentences)-1):
            chunk_left = sentences[max(i-w,0):i]
            chunk_right = sentences[i:min(i+w, len(sentences)-1)]
            v = self._get_topic_dist(' '.join(chunk_left))
            u = self._get_topic_dist(' '.join(chunk_right))
            c.append(self._coherence(u,v))
        
        # perform the smoothing (MA smoothing)
        cs = np.array(c)
        for g in range(1,s+1):
            hr = c[g:] + [c[-1]]*g # right side
            hl = [c[0]]*g + c[:-g] # left side
            cs += np.array(hr) + np.array(hl)
        cs = cs/(2*s+1)

        # find local extrema, candidates for splitting
        candidates = argrelmin(cs)[0]

        # build the scores
        prev = 0
        scores = []
        for i in range(candidates.size):
            left = np.max(cs[prev:candidates[i]])
            if i != candidates.size-1:
                #normal computation
                right = np.max(cs[candidates[i]:candidates[i+1]])
            else:
                #right is last point
                right = np.max(cs[candidates[i]:])
            scores.append(agg(left-cs[candidates[i]],right-cs[candidates[i]]))
            # update the previous candidate
            prev = candidates[i]
        scores = np.array(scores)
        
        # thresholding
        mean_score, std_score = np.mean(scores), np.std(scores)
        valleys = candidates[scores>(mean_score-k*std_score)]
        if diagnostics:
            plt.hist(scores, bins=20)
            plt.axvline(x=mean_score-k*std_score)
            plt.show()
            print(scores)
        return valleys, cs

    def split_text(self, document, paragraph_segment=False, **kwargs):
        if paragraph_segment and len(re.split(r'\n *\n', document))>1:
            sentences = []
            for paragraph in re.split(r'\n *\n', document):
                sentences.extend(sent_tokenize(paragraph))
                if sentences:
                    sentences[-1] += '\n\n'
        else:
            sentences = custom_tknzr.tokenize(document)
        # find the valleys
        valleys, cs = self._tile(sentences, **kwargs)
        # create the slicing list (split at separator 1 corresponds to indexing up to 2)
        slicing = [0] + (valleys + 1).tolist()+ [len(sentences)]
        topic_chunks = [' '.join(sentences[s:e]) for s,e in zip(slicing, slicing[1:])]
        return topic_chunks

# some hacking to keep the newlines even when using nltk sent_tokenizer (curtesy to HugoMailhot 
# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer)

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