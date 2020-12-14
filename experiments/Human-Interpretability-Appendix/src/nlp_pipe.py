# nlp_pipe.py
# contains the NLPPipe class for tokenization and preprocessing
import spacy
import spacy.lang.en
import gensim
import os


class NLPPipe:
    def __init__(self, max_freq = 0.5, min_wordcount = 5):
        self.__STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
        self.__trained = False
        self.nlp_spacy = spacy.load('en')
        self.__max_freq = max_freq
        self.__min_wordcount = min_wordcount
    def fit_transform(self, texts):
        '''

        Train The Transformer Transform the Texts to BOW. No real training is involved, the BOW dictionary is initialized.
        Attention: Texts must be passed as a list of texts, even if only one text is present
        '''
        # Tokenize
        docs = self.__texts2tokens(texts)
        # bigrammer
        self.__bigram = gensim.models.phrases.Phrases(docs, min_count=15)
        # add bigrams to bow
        for idx in range(len(docs)):
            for token in self.__bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
        # create dictionary
        self.dictionary = gensim.corpora.Dictionary(docs)
        # filter dictionary based on word frequency
        self.dictionary.filter_extremes(no_below=self.__min_wordcount, no_above=self.__max_freq)
        # transform the tokens to BOW
        corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        # set trained flag
        self.__trained = True
        return corpus
    
    def transform(self, texts):
        '''
        Transform the Texts to BOW.
        Attention: Texts must be passed as a list of texts, even if only one text is present
        '''
        # make sure the transformer has been trained
        if not self.__trained:
            raise RuntimeError('The transformer has not been trained yet')
        # Tokenize
        docs = self.__texts2tokens(texts)
        # add bigrams to bow
        for idx in range(len(docs)):
            for token in self.__bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
        # transform the tokens to BOW
        corpus = [self.dictionary.doc2bow(doc) for doc in docs]
        return corpus
    
    def save(self, folder):
        self.__bigram.save(os.path.join(folder, 'phrases'))
        self.dictionary.save(os.path.join(folder, 'dict'))

    def load(self, folder):
        self.__bigram = gensim.models.phrases.Phrases.load(os.path.join(folder, 'phrases'))
        self.dictionary = gensim.corpora.Dictionary.load(os.path.join(folder, 'dict'))
        self.__trained = True
        
    def __texts2tokens(self, texts):
        '''
        From Texts create token representation
        '''
        docs = list()

        for doc in self.nlp_spacy.pipe(texts, n_threads=5, batch_size=10):
            # Process document using Spacy NLP pipeline.
            ents = doc.ents  # Named entities

            # Keep only words (no numbers, no punctuation).
            # Lemmatize tokens, remove punctuation and remove stopwords.
            doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

            # Remove common words from a stopword list and keep only words of length 3 or more.
            doc = [token for token in doc if token not in self.__STOPWORDS and len(token) > 2]

            # Add named entities, but only if they are a compound of more than word.
            doc.extend([str(entity) for entity in ents if len(entity) > 1])

            docs.append(doc)
        return docs