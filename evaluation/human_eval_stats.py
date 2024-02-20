import os
import argparse
import statistics
import pandas as pd
import scipy.stats as stats


HUMAN_EVALUATION_METRICS = ["recall", "precision", "faithfulness"]
MODELS = ["pegasus", "pegasus_prunepert"]


def fill_worse_better(better_dict, worse_dict, couple_order, metric_value):
    first_model = couple_order.split("$")[0]
    second_model = couple_order.split("$")[1]

    if metric_value == 1:
        better_dict[first_model].append(1)
        worse_dict[first_model].append(0)

        better_dict[second_model].append(0)
        worse_dict[second_model].append(1)
    elif metric_value == 2:
        better_dict[first_model].append(0)
        worse_dict[first_model].append(1)

        better_dict[second_model].append(1)
        worse_dict[second_model].append(0)
    else:
        better_dict[first_model].append(0)
        worse_dict[first_model].append(0)

        better_dict[second_model].append(0)
        worse_dict[second_model].append(0)

    return better_dict, worse_dict


def compute_win_lose(metric_df, couple_orders, metric):
    better_dict = {}
    worse_dict = {}
    for model in MODELS:
        better_dict[model] = []
        worse_dict[model] = []

    for metric_value, couple_order in zip(metric_df[metric], couple_orders):
        fill_worse_better(better_dict, worse_dict, couple_order, metric_value)

    for model in MODELS:
        perc_win = sum(better_dict[model]) / len(better_dict[model])
        perc_lose = sum(worse_dict[model]) / len(worse_dict[model])
        print(f"{model} {metric} Win - Lose (%): {round(100 * (perc_win - perc_lose), 2)}")


def compute_kendall_tau(df_eval_1, df_eval_2):
    tau_list = []
    for metric in HUMAN_EVALUATION_METRICS:
        tau_list.append(stats.kendalltau(df_eval_1[metric], df_eval_2[metric])[0])

    avg_tau = statistics.mean(tau_list)

    return avg_tau


def kendall_tau_stats(dataframes):
    tau_list = []
    for i, df_eval_1 in enumerate(dataframes):
        for j, df_eval_2 in enumerate(dataframes):
            if i < j:
                avg_tau = compute_kendall_tau(df_eval_1, df_eval_2)
                print(f"Average tau between evaluator {i} and {j}: {avg_tau}")
                tau_list.append(avg_tau)

    overall_avg_tau = statistics.mean(tau_list)
    print(f"Overall average Tau: {overall_avg_tau}")


def direct_comparison(eval_files, df_keys):
    dataframes = []
    for file in eval_files:
        file_path = os.path.join("human_eval", file)
        dataframes.append(pd.read_excel(file_path)[:5])

    concatenated_df = pd.concat(dataframes, ignore_index=True)

    for metric in HUMAN_EVALUATION_METRICS:
            compute_win_lose(concatenated_df, df_keys["model_pairs"], metric)


def score_comparison(eval_files):
    dataframes = []
    for model in MODELS:
        for file in eval_files:
            file_path = os.path.join("human_eval", file)
            dataframes.append(pd.read_excel(file_path, sheet_name=model)[:5])

        concatenated_df = pd.concat(dataframes, ignore_index=True)
        for metric in HUMAN_EVALUATION_METRICS:
            print(f"Average {model} {metric} score: {statistics.mean(concatenated_df[metric])}")


def main():
    eval_files = [file for file in os.listdir("human_eval") if args.eval_file in file and "~$" not in file]

    if args.comparison == "direct":
        df_keys = pd.read_csv(os.path.join("human_eval", f"{args.summary}_direct_secret_keys.csv"))
        direct_comparison(eval_files, df_keys)
    else:
        score_comparison(eval_files)

        # kendall_tau_stats(dataframes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file",
        type=str,
    )
    parser.add_argument(
        "--summary",
        type=str,
    )
    parser.add_argument(
        "--comparison",
        type=str,
    )
    args = parser.parse_args()

    main()

