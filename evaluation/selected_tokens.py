import sys
sys.path.append('./')

import rouge
import torch
import argparse
import jsonlines
from statistics import mean
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import AutoTokenizer
from BARTScore.bart_score import BARTScorer


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


def get_selected_text(list_of_dicts, tokenizer):
    selected_texts = []
    not_selected_texts = []
    for data_dict in (list_of_dicts):
        input_ids = torch.tensor(data_dict['input_ids'][0])
        selected_indices = torch.tensor(data_dict["indicies"])
        all_indices = torch.arange(input_ids.size(0))

        selected_input_ids = torch.gather(input_ids, dim=0, index=selected_indices)
        selected_decoded_text = tokenizer.decode(selected_input_ids)

        not_selected_indices = torch.masked_select(all_indices,
                                                   torch.logical_not(torch.isin(all_indices, selected_indices)))
        not_selected_input_ids = torch.gather(input_ids, dim=0, index=not_selected_indices)
        not_selected_decoded_text = tokenizer.decode(not_selected_input_ids)

        selected_texts.append(selected_decoded_text)
        not_selected_texts.append(not_selected_decoded_text)

    return selected_texts, not_selected_texts


def compute_metrics(predictions, targets):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    result = {}
    rouge_scores = global_rouge_scorer.get_scores(references=targets, hypothesis=predictions)
    result["rouge-1"] = rouge_scores["rouge-1"]["f"]
    result["rouge-2"] = rouge_scores["rouge-2"]["f"]
    result["rouge-l"] = rouge_scores["rouge-l"]["f"]

    bartr_scores = bart_scorer.score(predictions, targets)
    bartp_scores = bart_scorer.score(targets, predictions)

    bart_score_R = mean(bartr_scores)
    bart_score_P = mean(bartp_scores)
    bart_score_F = mean([mean([pscore, rscore]) for pscore, rscore in zip(bartp_scores, bartr_scores)])
    result["bart-precision"] = bart_score_P
    result["bart-recall"] = bart_score_R
    result["bart-f1"] = bart_score_F


def calculate_stopword_percentage(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    num_stopwords = sum(1 for word in words if word.lower() in stop_words)
    total_words = len(words)
    percentage = (num_stopwords / total_words) * 100
    return percentage


def main():
    tokenizer = AutoTokenizer.from_pretrained("ccdv/lsg-pegasus-large-4096")

    list_of_dicts = []
    # Open the JSONL file and read lines
    with jsonlines.open(f'evaluation/cache/indicators_{args.summary}.jsonl') as reader:
        for obj in reader:
            list_of_dicts.append(obj)

    test_dataset = load_dataset(
        "paniniDot/sci_lay",
        "all",
    )["test"]

    targets = [target for target in test_dataset[f"{args.summary}_text"]]
    full_texts = [full_text for full_text in test_dataset["full_text"]]
    selected_texts, not_selected_texts = get_selected_text(list_of_dicts, tokenizer)
    
    print(f"[{args.summary} summarization]")
    print("Selected Texts")
    compute_metrics(selected_texts, targets)
    print("-----------")
    print("Not Selected Texts")
    compute_metrics(not_selected_texts, targets)
    print("-----------")
    print("Full Texts")
    compute_metrics(full_texts, targets)
    print("-----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train transformer to classify relevant tokens in dialog summarization"
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    main()