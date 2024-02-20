import os
import math
import evaluate
import torch
import rouge
import json
import numpy as np
from statistics import mean
from codecarbon import EmissionsTracker
from BARTScore.bart_score import BARTScorer
from nltk.translate.bleu_score import corpus_bleu


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


def get_carburacy(score, emission_train, emission_test, alpha=10, beta_train=1, beta_test=100):
    carburacy_train = None
    if emission_train is not None:
        carburacy_train = math.exp(math.log(score/100, alpha)) / (1 + emission_train * beta_train)
        carburacy_train = round(100 * carburacy_train, 2)
    carburacy_test = None
    if emission_test is not None:
        carburacy_test = math.exp(math.log(score/100, alpha)) / (1 + emission_test * beta_test)
        carburacy_test = round(100 * carburacy_test, 2)
    carburacy = None
    if carburacy_train is not None and carburacy_test is not None:
        carburacy = (2 * carburacy_train * carburacy_test) / (carburacy_train + carburacy_test)
        carburacy = round(100 * carburacy, 2)
    return carburacy_train, carburacy_test, carburacy


def predict(trainer, predict_dataset, max_predict_samples, training_args, tokenizer, train_emissions, split):
    test_tracker = EmissionsTracker(measure_power_secs=100000, save_to_file=False)
    test_tracker.start()
    predict_results = trainer.predict(predict_dataset, metric_key_prefix=split)
    test_emissions = test_tracker.stop()

    metrics = predict_results.metrics

    metrics[f"{split}_samples"] = min(max_predict_samples, len(predict_dataset))
    metrics[f"{split}_emissions"] = test_emissions

    if training_args.do_train:
        train_carburacy, predict_carburacy, carburacy = get_carburacy(metrics[f"{split}_R"], 
                                                                    train_emissions, test_emissions/len(predict_dataset))
        metrics["train_carburacy"] = train_carburacy
        metrics[f"{split}_carburacy"] = predict_carburacy
        metrics["carburacy"] = carburacy

    trainer.log_metrics(split, metrics)
    trainer.save_metrics(split, metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            list_output_dict = []
            for i, pred in enumerate(predictions):
                output_dict = {"prediction": pred}
                list_output_dict.append(output_dict)

            if training_args.new_dir is not None:
                if not os.path.exists(training_args.new_dir):
                    os.makedirs(training_args.new_dir)
                output_prediction_file = os.path.join(training_args.new_dir, f"generated_{split}_set.json")
            else:
                output_prediction_file = os.path.join(training_args.output_dir, f"generated_{split}_set.json")
            

            with open(output_prediction_file, 'w') as json_file:
                json.dump(list_output_dict, json_file, indent=4) 


def compute_metrics(references, predictions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_bertscore = evaluate.load("bertscore", cache_dir="cache_metric")
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')

    rouge_scores = global_rouge_scorer.get_scores(hypothesis=predictions, references=references)
    result = {"rouge1": round(100 * rouge_scores["rouge-1"]["f"], 2),
                "rouge2": round(100 * rouge_scores["rouge-2"]["f"], 2),
                "rougeL": round(100 * rouge_scores["rouge-l"]["f"], 2),
            }
    
    # Compute BLEU scores
    tokenized_predictions = [prediction.split(" ") for prediction in predictions]
    tokenized_labels = [[ref.split(" ")] for ref in references]

    result["bleu1"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1, 0, 0, 0)), 2)
    result["bleu2"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/2, 1/2, 0, 0)), 2)
    result["bleu3"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/3, 1/3, 1/3, 0)), 2)
    result["bleu4"] = round(100 * corpus_bleu(tokenized_labels, tokenized_predictions, weights=(1/4, 1/4, 1/4, 1/4)), 2)
    

    result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
        (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

    result_bs = metric_bertscore.compute(predictions=predictions, references=references, lang="en",
                                            idf=True, rescale_with_baseline=True,
                                            model_type="bert-base-uncased")
    result["bertscore"] = round(sum(result_bs["f1"]) / len(result_bs["f1"]) * 100, 2)

    bartr_scores = bart_scorer.score(predictions, references)
    bartp_scores = bart_scorer.score(references, predictions)

    bart_score_R = mean(bartr_scores)
    bart_score_P = mean(bartp_scores)
    bart_score_F = mean([mean([pscore, rscore]) for pscore, rscore in zip(bartp_scores, bartr_scores)])
    result["bart_score_R"] = round(bart_score_R, 3)
    result["bart_score_P"] = round(bart_score_P, 3)
    result["bart_score_F"] = round(bart_score_F, 3)
    
    return result
