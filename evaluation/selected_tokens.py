import sys

sys.path.append('./')

import time
import spacy
import rouge
import torch
import argparse
import jsonlines
from tqdm import tqdm
from statistics import mean
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

import plotly.graph_objects as go
import plotly.express as px

# from BARTScore.bart_score import BARTScorer

# "NUM", "PRON", "SPACE", "AUX"
MAIN_CONTENT_W = ["NOUN", "PUNCT", "VERB", "ADJ", "ADP", "STOPW", "DET"]
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
    # bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    result = {}
    rouge_scores = global_rouge_scorer.get_scores(references=targets, hypothesis=predictions)
    result["rouge-1"] = rouge_scores["rouge-1"]["f"]
    result["rouge-2"] = rouge_scores["rouge-2"]["f"]
    result["rouge-l"] = rouge_scores["rouge-l"]["f"]

    print(result)

    """
    bartr_scores = bart_scorer.score(predictions, targets)
    bartp_scores = bart_scorer.score(targets, predictions)

    bart_score_R = mean(bartr_scores)
    bart_score_P = mean(bartp_scores)
    bart_score_F = mean([mean([pscore, rscore]) for pscore, rscore in zip(bartp_scores, bartr_scores)])
    result["bart-precision"] = bart_score_P
    result["bart-recall"] = bart_score_R
    result["bart-f1"] = bart_score_F
    """


def calculate_stopword_percentage(texts):
    percentages = []
    for text in texts:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        num_stopwords = sum(1 for word in words if word.lower() in stop_words)
        total_words = len(words)
        percentage = (num_stopwords / total_words) * 100
        percentages.append(percentage)

    print(f"Stopword percentage: {mean(percentages)}")

    return mean(percentages)


def remove_stopwords(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]

    # Join the filtered words to form a sentence
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def calculate_percentage(elements):
    total_count = len(elements)
    element_count = {}
    for element in elements:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1

    percentages = {}
    for element, count in element_count.items():
        percentages[element] = (count / total_count) * 100

    return percentages


def print_percentages(percentages):
    for element, percentage in percentages.items():
        print(f"{element}: {percentage:.2f}%")


def normalize_list(numbers):
    min_val = min(numbers)
    max_val = max(numbers)
    normalized = [(x - min_val) / (max_val - min_val) for x in numbers]
    return normalized


def content_words(texts):
    nlp = spacy.load("en_core_sci_sm")

    content_ws = []
    entities_num = []
    for text in tqdm(texts):
        processed_text = nlp(text)
        entities_num.append(len(processed_text.ents))
        for token in processed_text:
            content_ws.append(token.pos_)

    num_entities = 100 * mean(normalize_list(entities_num))
    print(f"Average normalized number of entities: {num_entities}")
    percentages = calculate_percentage(content_ws)
    # percentages["NENTS"] = num_entities
    print_percentages(percentages)

    return percentages


def update_perc(percentages):
    # Calculate the sum of percentages for keys not in selected_keys
    other_percentage = sum(value for key, value in percentages.items() if key not in MAIN_CONTENT_W)

    # Create the new dictionary
    new_percentages = {key: value for key, value in percentages.items() if key in MAIN_CONTENT_W}
    new_percentages["OTHER"] = other_percentage

    return new_percentages


def plot_content_w(notseltok_cw, seltok_cw):
    figure = f"evaluation/cache/{args.summary}_content_w.pdf"
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(figure, format="pdf")
    time.sleep(2)

    fig = go.Figure()
    fig.add_trace(go.Histogram(histfunc="sum", y=list(seltok_cw.values()),
                               x=list(seltok_cw.keys()), name="Selected",
                               marker_color="rgb(102,194,165)"))
    fig.add_trace(go.Histogram(histfunc="sum", y=list(notseltok_cw.values()),
                               x=list(notseltok_cw.keys()), name="Not-selected",
                               marker_color="rgb(141,160,203)"))

    fig.update_layout(
        # showlegend=True if args.summary == "plain" else False,
        font=dict(size=15, family='latex'),
        legend=dict(
            orientation="h",
            yanchor="top",
            xanchor="center",
            x=0.5,
            y=1.12,
            font=dict(size=20),
        ))

    # Add a title above the radar charts using annotation
    fig.update_layout(
        annotations=[
            dict(
                text="<b>Lay Summarization</b>" if args.summary == "plain" else "<b>Technical Summarization</b>",
                # Title text
                x=0.5,  # X-coordinate: centered
                y=1.2,  # Y-coordinate: above the radar charts
                xref='paper',  # x-coordinate's reference point
                yref='paper',  # y-coordinate's reference point
                showarrow=False,  # No arrow
                font=dict(size=25),  # Title font size
            )
        ]
    )

    print("image writing")
    fig.write_image(figure, format="pdf")


def add_distribution(list_of_dicts, fig, type, marker_color):
    positions = [data["indicies"] for data in list_of_dicts]
    positions = [item for sublist in positions for item in sublist]
    normalized_positions = normalize_list(positions)

    fig.add_trace((go.Histogram(x=normalized_positions, marker_color=marker_color, name=type)))

    return fig


def token_position_distribution(plain_list_of_dicts, tech_list_of_dicts):
    fig = go.Figure()
    fig = add_distribution(plain_list_of_dicts, fig, "Lay Summarization", "rgb(102,194,165)")
    fig = add_distribution(tech_list_of_dicts, fig, "Technical Summarization", "rgb(141,160,203)")

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    fig.update_layout(
        yaxis_title="Frequency",
        font=dict(size=20, family='latex'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="center",
            x=0.5,
            y=1.02,
            font=dict(size=22),
        ))

    fig.show()

    fig.write_image(f"evaluation/cache/token_distribution.pdf", format="pdf")


def plot_positions():
    plain_list_of_dicts = []
    with jsonlines.open(f'evaluation/cache/indicators_plain.jsonl') as reader:
        for obj in reader:
            plain_list_of_dicts.append(obj)

    tech_list_of_dicts = []
    with jsonlines.open(f'evaluation/cache/indicators_technical.jsonl') as reader:
        for obj in reader:
            tech_list_of_dicts.append(obj)

    token_position_distribution(plain_list_of_dicts, tech_list_of_dicts)


def scores_correlation():

    list_of_dicts = []
    with jsonlines.open("/Users/paoloitaliani/Desktop/scores.jsonl") as reader:
        for obj in reader:
            list_of_dicts.append(obj)

    list_of_dicts = list_of_dicts[:100]
    scorer_scores = []
    for data in list_of_dicts:
        scorer_scores.extend(data["scorer_scores"])

    average_attn_scores = []
    for data in list_of_dicts:
        average_attn_scores.extend(data["average_attn_scores"])
    fig = px.scatter(x=normalize_list(scorer_scores), y=normalize_list(average_attn_scores))

    fig.show()


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

    if args.remove_stopwords:
        full_texts = [remove_stopwords(target) for target in full_texts]
        selected_texts = [remove_stopwords(pred) for pred in selected_texts]
        not_selected_texts = [remove_stopwords(pred) for pred in not_selected_texts]
        targets = [remove_stopwords(target) for target in targets]

    print(f"[{args.summary} summarization]")
    print("Selected Texts")
    seltok_cw = content_words(selected_texts)
    seltok_cw = update_perc(seltok_cw)
    # compute_metrics(selected_texts, targets)
    seltok_stp = calculate_stopword_percentage(selected_texts)
    seltok_cw["STOPW"] = seltok_stp
    print("-----------")
    print("Not Selected Texts")
    notseltok_cw = content_words(not_selected_texts)
    notseltok_cw = update_perc(notseltok_cw)
    # compute_metrics(not_selected_texts, targets)
    not_seltok_stp = calculate_stopword_percentage(not_selected_texts)
    notseltok_cw["STOPW"] = not_seltok_stp
    print("-----------")
    print("Full Texts")
    # compute_metrics(full_texts, targets)
    print("-----------")

    plot_content_w(notseltok_cw, seltok_cw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train transformer to classify relevant tokens in dialog summarization"
    )
    parser.add_argument(
        "--summary",
        type=str,
    )
    parser.add_argument(
        "--remove_stopwords",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    main()
