import torch
import rouge
import json
import argparse
import evaluate
from tqdm import tqdm
from statistics import mean
from datasets import load_dataset
from BARTScore.bart_score import BARTScorer

global_rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                  max_n=4,
                                  limit_length=True,
                                  length_limit=100,
                                  length_limit_type='words',
                                  apply_avg=True,
                                  apply_best=False,
                                  alpha=0.5,  # Default F1_score
                                  weight_factor=1.2,
                                  stemming=True)


def main():
    split_dataset = load_dataset(args.dataset_name, args.dataset_subset)[args.split]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    metric_bertscore = evaluate.load("bertscore")

    with open(f'{args.output_dir}/{args.model}/generated_{args.split}_set.json', 'r') as file:
        data = json.load(file)
    
    full_targets = [target for target in split_dataset[args.target_column]]
    full_predictions = [pred['prediction'] for pred in data]
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
        "--target_column",
        default="plain_text",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
    )
    args = parser.parse_args()

    main()
