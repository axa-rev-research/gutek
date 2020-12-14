# e4_compare segmenters.py
# compare the different segmentation methods
import argparse
from math import ceil
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from joblib import load
import random
from sklearn.metrics import precision_score, recall_score, jaccard_score

from nltk import sent_tokenize, word_tokenize

from src import topictiling, nlp_feature_extraction, htmm_segmenter, dataloader, text_interpreters

import sys
sys.path.append('src/nn_segmentation')
from neural_segmentation import NeuralSegmenter
def my_precision_score(y_true, y):
    try:
        return precision_score(y_true, y)
    except:
        return float('NaN')
    
def my_recall_score(y_true, y):
    try:
        return recall_score(y_true, y)
    except:
        return float('NaN')
    
def my_jaccard_score(y_true, y):
    try:
        return jaccard_score(y_true, y)
    except:
        return float('NaN')

def custom_sent_tokenize(document):
    if len(re.split(r'\n *\n', document))>1:
        sentences = []
        for paragraph in re.split(r'\n *\n', document):
            sentences.extend(sent_tokenize(paragraph))
            if sentences:
                sentences[-1] += '\n\n'
        return sentences
    else: 
        return sent_tokenize(document)

def perform_experiment(n_tries_per_segmenter, dataset, dataset_name, classifier, desired_change=0.07, do_htmm=True):
    print('Loading TopicTiling..')
    full_extractor = nlp_feature_extraction.NLPFeatureExtraction(model= ('Lda', 100), load = '../models/{}'.format(dataset_name))
    tiling = topictiling.TopicTiling(topictiling.Coherence(method=topictiling.METHOD_COSINE_SIMILARITY), full_extractor)
    # htmm
    if do_htmm:
        print('Loading HTMM..')
        htmm_sg = htmm_segmenter.HTMMSegmenter('../models/{}/dict_htmm.p'.format(dataset_name), '../models/{}/htmm.p'.format(dataset_name))
    # neural segmentation
    print('Loading NeuralSegmentation..')
    #model_path_fine = '../models/neural_segmentation/model_paragraph.t7'
    model_path_coarse = '../models/neural_segmentation/model_cpu.t7'
    word2vec_path = '../models/neural_segmentation/word2vec-google-news-300.gz'
    segmenter_coarse = NeuralSegmenter(model_path_coarse, word2vec_path)
    #segmenter_fine = NeuralSegmenter(model_path_fine, word2vec_path)
    print('Done, starting analysis.')

    segment_paragraph = lambda text: text.strip().split('\n\n')
    segment_sentence = lambda text: custom_sent_tokenize(text)
    segment_htmm = lambda text: htmm_sg.segment(text, paragraph_segment=True)
    segment_topictiling = lambda text: tiling.split_text(text, paragraph_segment=True)
    segment_neural = lambda text: segmenter_coarse(text, threshold=0.1)

    results = pd.DataFrame(columns=['method_gt', 'method_segment', 'ground_truth', 'prediciton', 'IoU', 'precision', 'recall'])

    for i in tqdm(range(n_tries_per_segmenter)):
        for gt_segmenter, method_gt in zip([segment_paragraph, segment_sentence, segment_topictiling, segment_neural, segment_htmm],
                                           ['segment_paragraph', 'segment_sentence', 'segment_topictiling', 'segment_neural', 'segment_htmm']):
            if (not do_htmm) and method_gt=='segment_htmm':
                # skip htmm if desired
                continue
            # create the new text
            target_class = np.random.randint(2)
            for i in range(50):
                # samle a donor and acceptor text
                acceptor = list(dataset.loc[dataset.target==target_class].text.sample(1))[0].strip()
                donor = list(dataset.loc[dataset.target!=target_class].text.sample(1))[0].strip()
                # segment the two texts
                chunks_acceptor = gt_segmenter(acceptor)
                chunks_donor = gt_segmenter(donor)
                # check if desired probability can be obtained
                desired_probability = classifier.predict_proba([acceptor])[0][target_class]-desired_change
                attained_probability = classifier.predict_proba([' '.join([acceptor, candidate]) for candidate in chunks_donor])[:, target_class]
                potential_inserts = [segment for segment, is_good in zip(chunks_donor, (attained_probability<=desired_probability).tolist()) if is_good]
                if potential_inserts:
                    insertion_point = np.random.randint(len(chunks_acceptor)+1)
                    inserted_chunk = random.sample(potential_inserts, 1)[0]
                    text = '\n\n'.join([c.replace('\n', ' ') for c in chunks_donor[:insertion_point]] + [inserted_chunk.replace('\n', ' ')] + [c.replace('\n', ' ') for c in chunks_donor[insertion_point:]])
                    true_prediction = np.array([0 for chunk in chunks_donor[:insertion_point] for s in custom_sent_tokenize(chunk)] + \
                        [1 for s in custom_sent_tokenize(inserted_chunk)] +\
                            [0 for chunk in chunks_donor[insertion_point:] for s in custom_sent_tokenize(chunk)])
                    break
            else:
                # no insert found, skip
                print('No insert found for {}'.format(method_gt))
                continue

            # test all segmentation methods
            for segmenter, method in zip([segment_paragraph, segment_sentence, segment_htmm, segment_topictiling, segment_neural],
                                         ['segment_paragraph', 'segment_sentence', 'segment_htmm', 'segment_topictiling', 'segment_neural']):

                if (not do_htmm) and method=='segment_htmm':
                    # skip htmm if desired
                    continue
                segments = segmenter(text)
                coefs = text_interpreters.interpret_regression(segments, lambda text: classifier.predict_proba([text])[0][target_class], 50)
                if len(coefs)>1:
                    best_coef = None
                    best_iou = -1
                    # find the best among the candidates
                    for bc in coefs[coefs<0]:
                        detection = np.array([int(c==bc) for segment, c in zip(segments, coefs) for sent in custom_sent_tokenize(segment)])
                        if my_jaccard_score(true_prediction, detection)>best_iou:
                            best_coef = bc
                            best_iou = my_jaccard_score(true_prediction, detection)
                    best_coef = best_coef if best_coef else min(coefs)
                else:
                    best_coef = coefs[0]
                
                
                detection = np.array([int(c==best_coef) for segment, c in zip(segments, coefs) for sent in custom_sent_tokenize(segment)])
                
                results = results.append(pd.DataFrame(data={
                    'method_gt' : [method_gt], 
                    'method_segment' : method, 
                    'ground_truth' : str(true_prediction), 
                    'prediciton' : str(detection), 
                    'IoU' : my_jaccard_score(true_prediction, detection), 
                    'precision' : my_precision_score(true_prediction, detection), 
                    'recall' : my_recall_score(true_prediction, detection)
                }))
    return results.reset_index(drop=True)
                
if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='On which dataset to train, must be one of [ng_transport, ng_religion, wiki, reuters, sentiment]')
    parser.add_argument('--n_tries', help='Number of samples to draw for each method (Default: 2)', default=2, type=int)
    args = parser.parse_args()

    #loading the data
    if args.dataset=='ng_transport':
        train, test = dataloader.load_20_newsgroups(clean=True, clean_test=True, dataset='transport')
    elif args.dataset=='ng_religion':
        train, test = dataloader.load_20_newsgroups(clean=True, clean_test=True, dataset='religion')
    elif args.dataset=='wiki':
        train, test = dataloader.load_wiki_city()
    elif args.dataset=='reuters':
        train, test = dataloader.load_reuters_news()
    elif args.dataset=='sentiment':
        train, test = dataloader.load_sentiment_dataset()
    else:
        raise Exception('The specified dataset does not exist')

    # load the model
    pipe = load('../models/{}/classifier.joblib'.format(args.dataset))
    # filtering the data
    test['prediction'] = pipe.predict(test.text)
    samples = test.loc[(test.prediction==test.target)&(test.text.str.len()>1000)].reset_index(drop=True)
    print(samples.groupby('target').count())
    results = perform_experiment(args.n_tries, samples, args.dataset,pipe, do_htmm=(args.dataset!='wiki'))
    results.to_csv('../results/e4/{}_v3.csv'.format(args.dataset))
    print(results.describe())