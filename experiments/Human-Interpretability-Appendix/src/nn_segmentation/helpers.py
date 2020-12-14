# helpers.py: Some helper functions for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_word_list(model, dictionary):
    words, probs = [], []
    for i in range(model.num_topics):
        topicterms = model.get_topic_terms(i)
        words.append([dictionary[t[0]] for t in topicterms])
        probs.append([t[1] for t in topicterms])

    fig, ax = plt.subplots(figsize=(model.num_topics*1.2, 5))
    sns.heatmap(pd.DataFrame(probs).T,
                annot=pd.DataFrame(words).T,
                fmt='',
                ax=ax,
                cmap='Blues',
                cbar=False)
    fig.tight_layout()



def show_word_list_lsi(model, dictionary):
    words, probs = [], []
    for i in range(model.num_topics):
        topicterms = model.show_topic(i)
        words.append([t[0] for t in topicterms])
        probs.append([t[1] for t in topicterms])

    fig, ax = plt.subplots(figsize=(model.num_topics*1.2, 5))
    sns.heatmap(pd.DataFrame(probs).T,
                annot=pd.DataFrame(words).T,
                fmt='',
                ax=ax,
                cmap='Blues',
                cbar=False)
    fig.tight_layout()

def show_word_list_hdp(model, dictionary):
    words, probs = [], []
    for i in range(model.get_topics().shape[0]):
        topicterms = model.show_topic(i)
        words.append([t[0] for t in topicterms])
        probs.append([t[1] for t in topicterms])

    fig, ax = plt.subplots(figsize=(model.get_topics().shape[0]*1.2, 5))
    sns.heatmap(pd.DataFrame(probs).T,
                annot=pd.DataFrame(words).T,
                fmt='',
                ax=ax,
                cmap='Blues',
                cbar=False)
    fig.tight_layout()

def show_coherence(model, corpus, tokens, top=10, cutoff=0.01):
    top_topics = model.top_topics(corpus=corpus, coherence='u_mass', topn=20)
    word_lists = pd.DataFrame(model.get_topics().T, index=tokens)
    order = []
    for w, word_list in word_lists.items():
        target = set(word_list.nlargest(top).index)
        for t, (top_topic, _) in enumerate(top_topics):
            if target == set([t[1] for t in top_topic[:top]]):
                order.append(t)

    fig, axes = plt.subplots(ncols=2, figsize=(15,5))
    title = f'# Words with Probability > {cutoff:.2%}'
    (word_lists.loc[:, order]>cutoff).sum().reset_index(drop=True).plot.bar(title=title, ax=axes[1]);

    umass = model.top_topics(corpus=corpus, coherence='u_mass', topn=20)
    pd.Series([c[1] for c in umass]).plot.bar(title='Topic Coherence', ax=axes[0])
    fig.tight_layout()

def print_highlight(strings, classes, highlight, sep=' '):
    '''
    Print string with highlighting classes.
    Cannot print more than 8 colors
    Parameters:
    strings : list of strings : the strings to print
    classes : list of ints : the classes for the strings
    highlight : dict (int to int) : color map for the classes, can only contain the values 0-7 
    '''
    if len(strings) != len(classes):
        raise ValueError('The lists \'strings\' and \'classes\' must have same length')
    for s, c in zip(strings, classes):
        if c == -1:
            # no class, print default
            print('\033[41m{}\033[m'.format(s), end=' ')
        else:
            if highlight[c] != 0:
                # background not black, print black text
                print('\033[4{}m{}\033[m'.format(highlight[c], s), end=sep)
            else:
                # background black, print white text
                print('\033[4{};37m{}\033[m'.format(highlight[c], s), end=sep)
    print("\n", end="")