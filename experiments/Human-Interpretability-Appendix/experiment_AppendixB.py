# e2_complexity.py
# perform experiment 2
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from nltk import sent_tokenize, word_tokenize

from src import topictiling, nlp_feature_extraction, htmm_segmenter, dataloader

import sys
sys.path.append('src/nn_segmentation')
from neural_segmentation import NeuralSegmenter

def evaluate_complexity(dataset, dataset_name, do_htmm=True, min_n_sentences=6):
    # load the models

    # topictiling
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

    # prepare the dataframe for the results
    results = pd.DataFrame(columns=['method', 'sample_id', 'dataset', 'comp_time', 'n_words', 'n_segments', 'words_per_segment',])

    # perform the analysis
    for sample_id, sample in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        # if the text is too short, skip the sample
        if len(sent_tokenize(sample.text.strip()))<min_n_sentences:
            continue
        # arrays for holding results
        method = ['paragraph', 'sentence', 'topictiling', 'neural']
        comp_time = []
        n_words = len(word_tokenize(sample.text.strip()))
        n_segments = []
        words_per_segment = []

        # perform anlaysis for paragraph segmentation
        start = time.time()
        segments = sample.text.strip().split('\n\n')
        comp_time.append(time.time()-start)
        n_segments.append(len(segments))
        words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))

        # perform anlaysis for sentence segmentation
        start = time.time()
        segments = sent_tokenize(sample.text.strip())
        comp_time.append(time.time()-start)
        n_segments.append(len(segments))
        words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))
        
        # perform analysis for topictiling (model loading outside computation time)
        start = time.time()
        segments = tiling.split_text(sample.text.strip(), s=1, k=1)
        comp_time.append(time.time()-start)
        n_segments.append(len(segments))
        words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))
        
        # perform analysis for fine neural method
        #start = time.time()
        #segments = segmenter_fine(sample.text.strip(), threshold=0.1)
        #comp_time.append(time.time()-start)
        #n_segments.append(len(segments))
        #words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))
        
        # perform analysis for coarse neural method
        start = time.time()
        segments = segmenter_coarse(sample.text.strip(), threshold=0.1)
        comp_time.append(time.time()-start)
        n_segments.append(len(segments))
        words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))
        
        # perform analysis for htmm
        if do_htmm:
            method.append('htmm')
            start = time.time()
            segments = htmm_sg.segment(sample.text.strip())
            comp_time.append(time.time()-start)
            n_segments.append(len(segments))
            words_per_segment.append(np.mean([len(word_tokenize(c)) for c in segments]))

        # log the results
        results = results.append(pd.DataFrame(data={
            'method' : method, 
            'sample_id' : sample_id,  
            'comp_time' : comp_time, 
            'n_words' : n_words, 
            'n_segments' : n_segments, 
            'words_per_segment' : words_per_segment,
        })).reset_index(drop=True)

    results.to_csv('../results/e2/{}.csv'.format(dataset_name))
    print('Results:')
    print('Average number of words: {}'.format(results.n_words.mean()))
    for method in results.method.unique():
        print('Segmentation Method {}:'.format(method))
        print('Number of words per segment: {}({})'.format(results.loc[results.method==method].words_per_segment.mean(),
                                                          results.loc[results.method==method].words_per_segment.std()))
        print('Number of segments: {}({})'.format(results.loc[results.method==method].n_segments.mean(),
                                                  results.loc[results.method==method].n_segments.std()))
        print('Computation Time: {}({})'.format(results.loc[results.method==method].comp_time.mean(),
                                                results.loc[results.method==method].comp_time.std()))
        print('\n\n')

if __name__ == "__main__":
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='On which dataset to train, must be one of [ng_transport, ng_religion, wiki, reuters, sentiment]')
    parser.add_argument('--do_htmm', help='Wether or not to also train the htmm', action='store_true')
    parser.add_argument('--min_n_sentences', help='Minimum number of sentences (Default: 6)', type=int, default=6)
    parser.add_argument('--out_file', help='File for the print output', default=None)
    args = parser.parse_args()

    # htmm failed to train on the wikipedia dataset
    if args.dataset=='wiki' and args.do_htmm:
        raise Exception('HTMM does not work for wiki dataset')
        
    if args.out_file:
        sys.stdout = open(args.out_file, "w+")

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
        train,test = dataloader.load_sentiment_dataset()
    else:
        raise Exception('The specified dataset does not exist')

    print('Performing Experiment:')
    print('Dataset: {}'.format(args.dataset))
    print('HTMM: {}'.format(args.do_htmm))
    print('Min number of sentences: {}'.format(args.min_n_sentences))
    # perform the experiment
    evaluate_complexity(test, args.dataset, do_htmm=args.do_htmm, min_n_sentences=args.min_n_sentences)