'''
interpreter.py
This file implements the overall loop for interpreting text results.
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.spatial.distance import cosine as cosine_distance
import statsmodels.api as sm

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt

from nltk import download, word_tokenize
from nltk.corpus import stopwords
import gensim
import torch

import sentence_embeddings
from models_infersent import InferSent


def get_text_html(text, alpha):
    '''
    add the background color to the text and return it as html
    '''
    if alpha<0:
        txt = '<span style="background-color:rgba(135,206,250,{})">{}</span>'.format(-alpha,text)
    else:
        txt = '<span style="background-color:rgba(255, 166, 0,{})">{}</span>'.format(alpha,text)
    return txt

def plt_colors(alpha):
    '''
    get the colors in RGB format
    '''
    if alpha<0:
        return (135/255,206/255,250/255, -alpha)
    else:
        return (255/255, 166/255, 0, alpha)

def display_text(chunks, alphas):
    '''
    Display the text with highlighting based on contribution
    '''
    htmls = []
    # build the html document
    for text, alpha in zip(chunks, alphas):
        htmls.append(get_text_html(text.replace('\n', ' <br> '), alpha))
    # display the html document
    display(HTML('<span class="tex2jax_ignore">'+' '.join(htmls)+'</span>'))
      
def display_importance(contribution, colors, uncertainty=None):
    '''
    Display the contribution as a bar chart
    '''
    plt.figure(figsize=(14,8))
    plt.barh(range(len(contribution)), contribution, color = colors, xerr=uncertainty)
    plt.gca().invert_yaxis()
    plt.xlabel('Contribution')
    plt.ylabel('Chunk index')
    plt.show()

def plot_importance(chunks, chunk_contributions, uncertainty = None, show_text=True):
    '''
    Plot the importance.
    Parameters:
        chunks : list of string : Chunks in the text
        chunk_contributions : nparray of float : contributions of the chunks to the result
    '''
    # calculate the correct alpha
    alpha = chunk_contributions/np.max(np.abs(chunk_contributions))
    # display the text interpretation
    if show_text:
        display_text(chunks, alpha)
    # display the chart interpretation
    colors = [plt_colors(a) for a in alpha]
    display_importance(chunk_contributions, colors, uncertainty=uncertainty)

class noise_representer_binary:
    def __call__(self, chunks_old, chunks_new):
        x = [int(c_old == c_new) for c_old, c_new in zip(chunks_old, chunks_new)]
        return x

class noise_representer_cos_sim:
    def __init__(self):
        self._embedder = sentence_embeddings.SentenceEmbedder()
    def __call__(self, chunks_old, chunks_new):
        x = []
        for c_old, c_new in zip(chunks_old, chunks_new):
            if c_old==c_new:
                x.append(1)
            else:
                try:
                    e_old, e_new = self._embedder.get_sentence_embeddings([c_old, c_new], ngram='unigrams', model='wiki')
                    x.append(1-cosine_distance(e_old, e_new))
                except AssertionError as ass_error:
                    x.append(0)
        return x

class noise_representer_LDA:
    def __init__(self, LDAModel):
        self._LDAModel = LDAModel
    def __call__(self, chunks_old, chunks_new):
        x = []
        for c_old, c_new in zip(chunks_old, chunks_new):
            if c_old==c_new:
                x.append(1)
            else:
                e_old = pd.DataFrame(data = self._LDAModel.predict_class(c_old), columns = ['Topic1', 'Probability1'])
                e_new = pd.DataFrame(data = self._LDAModel.predict_class(c_new), columns = ['Topic2', 'Probability2'])
                df = e_old.set_index('Topic1').join(e_new.set_index('Topic2'), how='inner')
                bhat = np.sqrt(df.Probability1*df.Probability2).sum()
                x.append(bhat)
        return x

class noise_representer_WMD:
    def __init__(self, model, gamma = 1):
        self.model = model
        download('stopwords')
        self.gamma = gamma

    def _preprocess(self, text):
        words = [w.lower() for w in word_tokenize(text) if w not in stopwords.words('english')]
        return words

    def __call__(self, chunks_old, chunks_new):
        x=[]
        for c_old, c_new in zip(chunks_old, chunks_new):
            if c_old==c_new:
                x.append(1)
            else:
                chunk1 = self._preprocess(c_old)
                chunk2 = self._preprocess(c_new)
                distance = self.model.wv.wmdistance(chunk1, chunk2)
                similarity = np.exp(-self.gamma*distance)
                x.append(similarity)
        return x

class noise_representer_infersent:
    def __init__(self, model_version=1, verbose = False):
        MODEL_PATH = "/Users/b054of/Documents/packages/encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self._verbose = verbose
        
        
        # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
        W2V_PATH = '/Users/b054of/Documents/packages/GloVe/glove.840B.300d.txt' \
                    if model_version == 1 else '/Users/b054of/Documents/packages/fastText/crawl-300d-2M.vec'
        self.model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        self.model.build_vocab_k_words(K=100000)
        
    def _cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    def __call__(self, chunks_old, chunks_new):
        x = []
        for c_old, c_new in zip(chunks_old, chunks_new):
            if c_old==c_new:
                x.append(1)
            else:
                similarity = self._cosine(self.model.encode([c_old])[0], self.model.encode([c_new])[0])
                if self._verbose and similarity<0:
                    print(similarity)
                x.append(similarity)
        return x

def importance_analysis(chunks, chunk_resample, prediction_function, sampler, noise_representer=noise_representer_binary, n_samples=100, p_binomial=0.7, seed = 2020, uncertainty=False, show_text=True):
    '''
    Perform importance analysis
    Parameters:
        chunks : list of string : The chunks of the text
        chunk_resample : function handle : how to "remove" a chunk, function maps from list of chunks to list of chunks
        prediction functiono : function handle : the function on which interpretability shall be applied
    '''
    X = []
    y = []
    if seed:
        np.random.seed(seed)
    
    X_sampled = sampler(chunks, n_samples)
    for x in X_sampled:
        chunks_input = chunk_resample(chunks, x)
        text = ' '.join(chunks_input)
        if len(chunks_input)==len(chunks):
            x = noise_representer(chunks, chunks_input)
        else:
            x = x
        X.append(x)
        y.append(prediction_function(text))

    
    if uncertainty:
        X_ = sm.add_constant(np.array(X))
        res = sm.OLS(y,X_).fit()
        coefficients = res.params[1:]
        uncertainty = 2*res.bse[1:]
        print('R2 of interpretability regression: {}'.format(res.rsquared))
    else:
        reg = LinearRegression().fit(np.array(X),np.array(y))
        coefficients = reg.coef_
        uncertainty = None
        print('R2 of interpretability regression: {}'.format(r2_score(y, reg.predict(np.array(X)))))

    plot_importance(chunks, coefficients, uncertainty=uncertainty, show_text=show_text)
    return coefficients, uncertainty, X
