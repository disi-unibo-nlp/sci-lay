import os
import json
import random
import argparse
import secrets
import string
from datasets import load_dataset
import pandas as pd


HUMAN_EVALUATION_METRICS = ["recall", "precision", "faithfulness"]
MODELS = ["pegasus", "pegasus_prunepert"]


def get_predictions(path):
    with open(path, 'r') as file:
        predictions = json.load(file)

    predictions = [predictions[i]["prediction"] for i in args.indices]
    return predictions[:args.num_instances]


def create_scores_file(inputs, targets, predictions, writer, model):

    df_evaluations = pd.DataFrame({"inputs": inputs, "targets": targets, "models_couples": predictions})

    for metric in HUMAN_EVALUATION_METRICS:
        df_evaluations[metric] = ["" for _ in range(len(df_evaluations))]
    df_evaluations.to_excel(writer, sheet_name=model, index=False)

    return writer


def create_syn_data_scores(inputs, qa_pairs, writer):
    qa_pairs = [pair.split("Answer:")[0] + "\n Answer:" + pair.split("Answer:")[1] for pair in qa_pairs]
    df_evaluations = pd.DataFrame({"inputs": inputs, "qa_pairs": qa_pairs})

    for metric in HUMAN_EVALUATION_METRICS:
        df_evaluations[metric] = ["" for _ in range(len(df_evaluations))]
    df_evaluations.to_excel(writer, index=False)

    return writer


def create_direct_comparison(inputs, targets, predictions, writer):
    model_pairs, model_pairs_keys, concatenated_pairs = create_models_pairs(predictions)

    df_evaluations = pd.DataFrame({"inputs": inputs, "targets": targets, "model_pairs": model_pairs,
                                   "model_pairs_keys": model_pairs_keys, "concatenated_pairs": concatenated_pairs})
    df_evaluations = df_evaluations.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create a new DataFrame with the columns to be removed
    df_secret = df_evaluations[["model_pairs_keys", "model_pairs"]].copy()

    # Remove the columns from df_evaluations
    df_evaluations.drop(columns=["model_pairs"], inplace=True)

    for metric in HUMAN_EVALUATION_METRICS:
        df_evaluations[metric] = ["" for _ in range(len(df_evaluations))]
    df_evaluations.to_excel(writer, index=False)

    df_secret.to_csv(f"human_eval/{args.summary}_{args.comparison}_secret_keys.csv", index=False)

    return writer


def generate_random_key(length):
    alphabet = string.ascii_letters + string.digits  # You can include other characters if needed
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return key


def create_models_pairs(predictions):
    # Generate all possible combinations of model pairs

    model_pairs = []
    model_pairs_keys = []
    concatenated_pairs = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):

            model1 = MODELS[i]
            model2 = MODELS[j]

            predictions_model1 = predictions[model1]
            predictions_model2 = predictions[model2]

            for pred1, pred2 in zip(predictions_model1, predictions_model2):
                if random.choice([True, False]):
                    model_pairs.append(f"{model1}${model2}")
                    model_pairs_keys.append(generate_random_key(10))
                    concatenated_pairs.append(f"{pred1}"
                                              f" ################## \n {pred2}")
                else:
                    model_pairs.append(f"{model2}${model1}")
                    model_pairs_keys.append(generate_random_key(10))
                    concatenated_pairs.append(f"{pred2}"
                                              f" ################## \n {pred1}")

    return model_pairs, model_pairs_keys, concatenated_pairs


def fill_xlsx_file(inputs, targets, writer):
    folder = f"output_{args.summary}"
    if args.comparison == "direct":
        predictions = {}
        for model in MODELS:
            predictions[model] = get_predictions(os.path.join(folder, f"{model}.json"))
        writer = create_direct_comparison(inputs, targets, predictions, writer)
    else:
        for model in MODELS:
            predictions = get_predictions(os.path.join(folder, f"{model}.json"))
            writer = create_scores_file(inputs, targets, predictions, writer, model)

    return writer


def main():
    column_names = {
        "input": "full_text",
        "target": f"{args.summary}_text",
    }
    # Output XLSX filename
    if not os.path.exists("human_eval"):
        os.makedirs("human_eval")
    output_xlsx = os.path.join("human_eval", f"{args.summary}_{args.comparison}.xlsx")

    dataset = load_dataset(args.dataset_name, args.dataset_subset)[args.split]
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    args.indices = indices
    # Apply the same shuffle order to both train_dataset and predictions
    dataset = [dataset[i] for i in indices][:args.num_instances]

    inputs = [instance[column_names["input"]] for instance in dataset]
    targets = [instance[column_names["target"]] for instance in dataset]

    # Create a Pandas Excel writer
    writer = pd.ExcelWriter(output_xlsx, engine="xlsxwriter")
    writer = fill_xlsx_file(inputs, targets, writer)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="paniniDot/sci_lay",
        type=str,
    )
    parser.add_argument(
        "--split",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset_subset",
        default="all",
        type=str,
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=50
    )
    parser.add_argument(
        "--comparison",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    main()

