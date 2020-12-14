# e3_context.py
# perform experiment 3

import argparse
import re, os

from nltk.tokenize import sent_tokenize
from joblib import load
from lime.lime_text import LimeTextExplainer

from src import (
    topictiling,
    nlp_feature_extraction,
    htmm_segmenter,
    dataloader,
    text_interpreters,
)

import sys

sys.path.append("src/nn_segmentation")
from neural_segmentation import NeuralSegmenter


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def explain_sample(text, dataset_name, sample_id, segmenter, n_samples):
    # segment the text
    if segmenter == "topictiling":
        full_extractor = nlp_feature_extraction.NLPFeatureExtraction(
            model=("Lda", 100), load="../models/{}".format(dataset_name)
        )
        tiling = topictiling.TopicTiling(
            topictiling.Coherence(method=topictiling.METHOD_COSINE_SIMILARITY),
            full_extractor,
        )
        segments = tiling.split_text(text)
    elif segmenter == "htmm":
        htmm_sg = htmm_segmenter.HTMMSegmenter(
            "../models/{}/dict_htmm.p".format(dataset_name),
            "../models/{}/htmm.p".format(dataset_name),
        )
        segments = htmm_sg.segment(text)
    elif segmenter == "neural":
        model_path_coarse = "../models/neural_segmentation/model_cpu.t7"
        word2vec_path = "../models/neural_segmentation/word2vec-google-news-300.gz"
        segmenter_ = NeuralSegmenter(model_path_coarse, word2vec_path)
        segments = segmenter_(text, threshold=0.1)
    elif segmenter == "paragraph":
        segments = [paragraph + "\n\n" for paragraph in text.split("\n\n")]
    elif segmenter == "sentence":
        segments = []
        for paragraph in text.split("\n\n"):
            segments.extend(sent_tokenize(paragraph))
            segments[-1] += "\n\n"
    else:
        raise Exception("Invalid segmentation method provided")

    # perform the interpretation
    model = load("../models/{}/classifier.joblib".format(dataset_name))
    coefs = text_interpreters.interpret_regression(
        segments, lambda text: model.predict_proba([text])[0][1], n_samples
    )
    # save the result
    ensure_dir("../results/e3/{}/{}/{}/".format(dataset_name, sample_id, segmenter))
    text_interpreters.plot_contribution(
        segments,
        coefs,
        path="../results/e3/{}/{}/{}/".format(dataset_name, sample_id, segmenter),
    )
    # visualize result in LIME format
    # text_interpreters.plot_contribution_LIME_style(segments, coefs)
    text_interpreters.plot_contribution_LIME_style(
        segments,
        coefs,
        path="../results/e3/{}/{}/{}/".format(dataset_name, sample_id, segmenter),
    )


if __name__ == "__main__":
    # parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="On which dataset to train, must be one of [ng_transport, ng_religion, wiki, reuters, sentiment]",
    )
    parser.add_argument(
        "--method",
        help="Segmentation_method, must be one of [topictiling, htmm, neural, paragraph, sentence, LIME]",
    )
    parser.add_argument("--sample_id", help="id of the sample", type=int)
    parser.add_argument(
        "--nsamples",
        help="Number of samples to draw from the local neighborhood (Default: 20)",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    if args.dataset == "wiki" and args.method == "htmm":
        raise Exception("HTMM does not work for wiki dataset")

    # loading the data
    if args.dataset == "ng_transport":
        train, test = dataloader.load_20_newsgroups(
            clean=True, clean_test=True, dataset="transport"
        )
    elif args.dataset == "ng_religion":
        train, test = dataloader.load_20_newsgroups(
            clean=True, clean_test=True, dataset="transport"
        )
    elif args.dataset == "wiki":
        train, test = dataloader.load_wiki_city()
    elif args.dataset == "reuters":
        train, test = dataloader.load_reuters_news()
    elif args.dataset == "sentiment":
        train, test = dataloader.load_sentiment_dataset()
    else:
        raise Exception("The specified dataset does not exist")

    text = test.text[args.sample_id]
    text = re.sub("(?<!\\n)\\n(?!\\n)", " ", text.replace("\t", "")).replace(
        "\n\n\n", "\n\n"
    )
    if args.method == "LIME":
        explainer = LimeTextExplainer(class_names=["-", "+"])
        classifier = load("../models/{}/classifier.joblib".format(args.dataset))
        exp = explainer.explain_instance(text, classifier.predict_proba)
        print(exp.as_list())
        ensure_dir(
            "../results/e3/{}/{}/{}/".format(args.dataset, args.sample_id, args.method)
        )
        exp.save_to_file(
            "../results/e3/{}/{}/{}/results.html".format(
                args.dataset, args.sample_id, args.method
            ),
            text=True,
        )
    else:
        explain_sample(text, args.dataset, args.sample_id, args.method, args.nsamples)
