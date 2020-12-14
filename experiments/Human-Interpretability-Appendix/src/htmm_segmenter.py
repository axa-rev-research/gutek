# htmm_segmenter.py
# segment the text using a pretrained htmm model and pyhtmm package
import re
import gensim
import pickle
import pyhtmm
import spacy
from nltk import sent_tokenize

class HTMMSegmenter:
    def __init__(self, dictpath, htmmpath, n_topics=120):
        self.dictionary = gensim.corpora.Dictionary.load(dictpath, htmmpath)
        self.htmm = pickle.load(open(htmmpath, "rb" ))
        self.nlp = nlp = spacy.load('en')
        self.n_topics = n_topics

    def segment(self, text, paragraph_segment=False):
        path, sents = self.infer_path(text, paragraph_segment=paragraph_segment)
        chunks, chunk_topics = [],[]
        for s, t in zip(sents, path):
            if chunk_topics and t==chunk_topics[-1]:
                chunks[-1] = chunks[-1] + ' ' + s
            else:
                chunks.append(s)
                chunk_topics.append(t)
        return chunks

    def infer_path(self, text, paragraph_segment=False):#, htmm, num_topics, dictionary):
        if paragraph_segment and len(re.split(r'\n *\n', text))>1:
            sentences = []
            for paragraph in re.split(r'\n *\n', text):
                sentences.extend(sent_tokenize(paragraph))
                if sentences:
                    sentences[-1] += '\n\n'
        else:
            sentences = sent_tokenize(text)
        # tokenize
        #sentences = sent_tokenize(text)
        doc = [[t.lemma_.lower() for t in self.nlp(s) if t.is_alpha and not t.is_stop] for s in sentences]
        # convert to dictionary index
        doc_idx = [[i for i in self.dictionary.doc2idx(s) if i!=-1] for s in doc]
        # convert to pyhtmm forma
        d = pyhtmm.document._Document()
        for sent in doc_idx:
            s = pyhtmm.sentence._Sentence('noneyves')
            for word in sent:
                s.add_word(word)
            d.add_sentence(s)
        if d.num_sentences == 0:
            raise Exception('After Preprocessing, the Document consists of 0 sentences')
        
        path, entropy = self.htmm.predict_topic(d)
        path_clean = [s if s<self.n_topics else s-self.n_topics for s in path]
        return path_clean, sentences