import sys
sys.path.append('./')

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import torch
import rouge
import json
import nltk
import argparse
import evaluate
import numpy as np
from tqdm import tqdm
from evaluation.compute_readability import print_read
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from bleurt import score as bleurt_scorer
from statistics import mean
from datasets import load_dataset
# from moverscore_folder.moverscore_v2 import get_idf_dict, word_mover_score 
from BARTScore.bart_score import BARTScorer
from readability import Readability
from datasets import load_dataset

METRICS = ["fkgl", "cli", "gf", "dcrs"]





global_rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                  max_n=4,
                                  limit_length=True,
                                  length_limit=500,
                                  length_limit_type='words',
                                  apply_avg=True,
                                  apply_best=False,
                                  alpha=0.5,  # Default F1_score
                                  weight_factor=1.2,
                                  stemming=True)


def evaluate_by_journal(split_dataset, full_targets, full_predictions, bart_scorer, metric_bertscore):
    data_dict = {}
    journals = set(split_dataset['journal'])
    for i, journal in enumerate(split_dataset['journal']):
        if journal not in data_dict:
            data_dict[journal] = {}
            data_dict[journal]["targets"] = []
            data_dict[journal]["predictions"] = []

        data_dict[journal]["targets"].append(full_targets[i])
        data_dict[journal]["predictions"].append(full_predictions[i])

    full_output_string = ""
    for journal in tqdm(journals):
        predictions = data_dict[journal]["predictions"]
        targets = data_dict[journal]["targets"]
        
        bartr_scores = bart_scorer.score(predictions, targets)
        bartp_scores = bart_scorer.score(targets, predictions)

        bart_score_R = mean(bartr_scores)
        bart_score_P = mean(bartp_scores)
        bart_score_F = mean([mean([pscore, rscore]) for pscore, rscore in zip(bartp_scores, bartr_scores)])
        
        result_bs = metric_bertscore.compute(predictions=predictions, references=targets, lang="en",
                                                idf=True, rescale_with_baseline=True,
                                                model_type="bert-base-uncased")
        bert_score_F1 = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)
        bert_score_r = round(sum(result_bs["recall"]) / len(result_bs["recall"]) * 100, 2)
        bert_score_p = round(sum(result_bs["precision"]) / len(result_bs["precision"]) * 100, 2)
        

        rouge_scores = global_rouge_scorer.get_scores(references=targets, hypothesis=predictions)
        print_command_as_string = '''\
        ------------------------------------------------------------------------------------------
        Journal: {journal} || Number of samples: {num_samples}
        \t\t ROUGE-1 \t ROUGE-2 \t ROUGE-L \t BERTScore \t BARTScore
        Recall: \t {rouge_1_r} \t\t {rouge_2_r} \t\t {rouge_l_r} \t\t {bert_score_r}  \t\t {bart_score_R}
        precision: \t {rouge_1_p} \t\t {rouge_2_p} \t\t {rouge_l_p} \t\t {bert_score_p} \t\t {bart_score_P}
        F-score: \t {rouge_1_f} \t\t {rouge_2_f} \t\t {rouge_l_f} \t\t {bert_score_F1} \t\t {bart_score_F}
        ------------------------------------------------------------------------------------------
        \n\n
        '''

        # Replace placeholders with actual values
        formatted_command = print_command_as_string.format(
            journal=journal,
            num_samples=len(predictions),
            rouge_1_r=round(100 * rouge_scores["rouge-1"]["r"], 2),
            rouge_2_r=round(100 * rouge_scores["rouge-2"]["r"], 2),
            rouge_l_r=round(100 * rouge_scores["rouge-l"]["r"], 2),
            bert_score_r=bert_score_r,
            bart_score_R=round(bart_score_R, 3),
            rouge_1_p=round(100 * rouge_scores["rouge-1"]["p"], 2),
            rouge_2_p=round(100 * rouge_scores["rouge-2"]["p"], 2),
            rouge_l_p=round(100 * rouge_scores["rouge-l"]["p"], 2),
            bert_score_p=bert_score_p,
            bart_score_P=round(bart_score_P, 3),
            rouge_1_f=round(100 * rouge_scores["rouge-1"]["f"], 2),
            rouge_2_f=round(100 * rouge_scores["rouge-2"]["f"], 2),
            rouge_l_f=round(100 * rouge_scores["rouge-l"]["f"], 2),
            bert_score_F1=bert_score_F1,
            bart_score_F=round(bart_score_F, 3)
        )

        full_output_string += formatted_command

    with open(f'{args.output_dir}/{args.model}/journals_results.txt', 'w') as file:
        file.write(full_output_string)


def remove_stopwords(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

    # Join the filtered words to form a sentence
    filtered_text = ' '.join(filtered_words)

    return filtered_text

def main():
    split_dataset = load_dataset(args.dataset_name, args.dataset_subset)[args.split]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    metric_bertscore = evaluate.load("bertscore")
    metric_rouge = evaluate.load("rouge")

    with open(f'{args.output_dir}/{args.model}/generated_{args.split}_set.json', 'r') as file:
        data = json.load(file)
    
    full_targets = [target for target in split_dataset[args.output_dir.split("output_")[1]]]
    full_predictions = [pred['prediction'] for pred in data]

    if args.remove_stopwords:
        full_targets = [remove_stopwords(target) for target in full_targets]
        full_predictions = [remove_stopwords(pred) for pred in full_predictions]

    if args.by_journal:
        evaluate_by_journal(split_dataset, full_targets, full_predictions, bart_scorer, metric_bertscore)
    else:
        result = {}
        """
        # MoverScore
        idf_dict_ref = get_idf_dict(full_targets)
        idf_dict_hyp = get_idf_dict(full_predictions)
        with open('moverscore_folder/examples/stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = set(f.read().strip().split(' '))
        scores = word_mover_score(full_targets, full_predictions, idf_dict_ref, idf_dict_hyp, \
                          stop_words=stop_words, n_gram=1, remove_subwords=True, batch_size=10)
        result["moverscore"] = np.mean(scores)
        """
        """
        # Bleurt
        scorer = bleurt_scorer.BleurtScorer()
        scores = scorer.score(references=full_targets, candidates=full_predictions)
        result["bleurt"] = np.mean(scores)
        
        # BertScore
        result_bs = metric_bertscore.compute(predictions=full_predictions, references=full_targets, lang="en",
                                                idf=True, rescale_with_baseline=True,
                                                model_type="distilbert-base-uncased")
        
        result["bertscore_pubmed"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)
        result = metric_rouge.compute(predictions=full_predictions, references=full_targets)
        
        
        full_predictions = " ".join(full_predictions)
        print_read(full_predictions)
        """

        rouge_scores = global_rouge_scorer.get_scores(references=full_targets, hypothesis=full_predictions)
        result["rouge-1"] = rouge_scores["rouge-1"]["f"]
        result["rouge-2"] = rouge_scores["rouge-2"]["f"]
        result["rouge-l"] = rouge_scores["rouge-l"]["f"]
             
        print(result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train transformer to classify relevant tokens in dialog summarization"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default="paniniDot/sci_lay",
        type=str,
    )
    parser.add_argument(
        "--dataset_subset",
        default="all",
        type=str,
    )
    parser.add_argument(
        "--split",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="output_plain_text",
        type=str,
    )
    parser.add_argument(
        "--by_journal",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--remove_stopwords",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    main()
