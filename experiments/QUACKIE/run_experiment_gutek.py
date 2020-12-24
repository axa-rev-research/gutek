# run_experiment.py
# Run experiments for custom interpreter

# You should define the interpreter function handle below the imports. It takes the model, question and context as input and returns the sentence attribution scores as output (high score=important)
# Trivial Example:
# >> from nltk.tokenize import sent_tokenize
# >> interpreter = lambda model, question, context: list(range(len(sent_tokenize(context))))
import argparse
import torch

import qa_experimenters
import models
import os

# IMPORT YOUR CODE HERE
# "interpreter" should be defined
# +----------------------------+
# |     CUSTOM CODE START      |
# +----------------------------+
from functools import partial
import gutek

interpreter = partial(gutek.gutek_interpreter, binary=False, n_samples=10, batch_size=2)
# +----------------------------+
# |     CUSTOM CODE END        |
# +----------------------------+


def run(dataset, device):
    try:
        interpreter
    except NameError:
        raise NotImplementedError("Interpreter not defined, check the top of the source code on how to use your model.")
    else:
        # run for QA model
        print('Running with QA model...')
        experiment = qa_experimenters.SQuADExperimenter(models.Model_QA(device=device), interpreter) if dataset == 'SQuAD' \
            else qa_experimenters.SQuADShiftsExperimenter(models.Model_QA(device=device), interpreter, dataset.lower())
        experiment.experiment()
        if not os.path.exists('results_QA_' + dataset):
            os.makedirs('results_QA_' + dataset)
        experiment.save(path='results_QA_' + dataset)
        print('Done!')

        # run for Classification model
        print('Running with Classification model...')
        experiment = qa_experimenters.SQuADExperimenter(models.Model_Classification(device=device), interpreter) if dataset == 'SQuAD' \
            else qa_experimenters.SQuADShiftsExperimenter(models.Model_Classification(device=device), interpreter, dataset.lower())
        experiment.experiment()
        if not os.path.exists('results_quackie/results_Classification_' + dataset):
            os.makedirs('results_quackie/results_Classification_' + dataset)
        experiment.save(path='results_quackie/results_Classification_' + dataset)
        print('Done!')


def analyze(args):
    E = qa_experimenters.QAExperimenter(None, None, None)
    df = df = pd.DataFrame(
        columns=[
            "model",
            "interpreter",
            "mean_iou",
            "mean_hpd",
            "mean_snr",
            "std_iou",
            "std_hpd",
            "std_snr",
            "fails",
            "no_snr",
            "info",
            "dataset",
        ]
    )
    for folder in os.listdir("results_quackie"):
        E.load("results_quackie/results_QA_SQuAD")
        res = E.analyze(interpreter)[
            [
                "mean_iou",
                "mean_hpd",
                "mean_snr",
                "std_iou",
                "std_hpd",
                "std_snr",
                "fails",
                "no_snr",
            ]
        ]
        res["model"] = folder.split("_")[1]
        res["dataset"] = folder.split("_")[2]
        res["interpreter"] = args.name
        res["info"] = args.info
        df = df.append(res)
    df = df[
        ["dataset", "model", "interpreter", "info", "mean_iou", "mean_hpd", "mean_snr"]
    ]
    df.columns = ["dataset", "classifier", "method", "info", "IoU", "HPD", "SNR"]
    print(df)
    df.to_json("results.json", orient="records")
    print("Saved results in results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters of the experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SQuAD", "NEW_WIKI", "NYT", "Reddit", "Amazon"],
        help="Dataset to use, can be one of ['SQuAD', 'NEW_WIKI', 'NYT', 'Reddit', 'Amazon'], SQuAD for SQuAD dataset, others for SQuADShifts",
    )
    parser.add_argument(
        "--run",
        dest="run",
        action="store_true",
        help="Flag if Experiment should be run",
    )
    parser.add_argument(
        "--analyze",
        dest="analyze",
        action="store_true",
        help="Flag if Experiment should be analyzed (all results will be analyzed).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Custom",
        help="Name of the interpreter to use in results",
    )
    parser.add_argument(
        "--info", type=str, default="Custom", help="Content for the 'Info' column",
    )
    parser.add_argument(
        "--no_cuda", dest="no_cuda", action="store_true", help="Flag if force cpu usage"
    )
    args = parser.parse_args()
    if not (args.run or args.analyze):
        print("Please run or analyze")

    else:
        device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        print("Using device {}".format(device))
        if args.run:
            assert args.dataset, "Dataset required for running"
            run(args.dataset, device)
        if args.analyze:
            print("\nAnalyzing for all datasets..")
            analyze(args)

        print(
            "\n\nThank you for choosing QUACKIE. You can submit your results via git pull request, more info here: "
        )
        print("https://github.com/axa-rev-research/quackie/tree/gh-pages")
