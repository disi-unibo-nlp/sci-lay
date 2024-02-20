import sys
sys.path.append('./')

from tqdm import tqdm
from datasets import load_dataset
from evaluation.fragments import Fragments


import nltk
from nltk import ngrams


# Function to compute n-grams from a given text
def get_ngrams(text, n):
    words = nltk.word_tokenize(text)
    n_grams = list(ngrams(words, n))
    return [' '.join(gram) for gram in n_grams]


# Function to compute the proportion of novel n-grams between two texts
def compute_novelty_proportion(text1, text2, n):
    ngrams_text1 = set(get_ngrams(text1, n))
    ngrams_text2 = set(get_ngrams(text2, n))

    novel_ngrams = ngrams_text2 - ngrams_text1

    proportion = len(novel_ngrams) / len(ngrams_text2)

    return proportion


def compute_abstractiveness(full_texts, summaries):

    compression_list = []
    coverage_list = []
    density_list = []
    uni_grams_list = []
    bi_grams_list = []
    tri_grams_list = []

    for full_text, summary in tqdm(zip(full_texts, summaries)):
        uni_grams_list.append(compute_novelty_proportion(full_text, summary, 1))
        bi_grams_list.append(compute_novelty_proportion(full_text, summary, 2))
        tri_grams_list.append(compute_novelty_proportion(full_text, summary, 3))

        # fragments = Fragments(summary, full_text)
        # density_list.append(fragments.density())
        # coverage_list.append(fragments.coverage())
        # compression_list.append(fragments.compression())

    # print(f"Average Compression: {sum(compression_list) / len(compression_list)}")
    # print(f"Average Coverage: {sum(coverage_list) / len(coverage_list)}")
    # print(f"Average Density: {sum(density_list) / len(density_list)}")
    print(f"Average uni-grams proportion: {sum(uni_grams_list)/len(uni_grams_list)}")
    print(f"Average bi-grams proportion: {sum(bi_grams_list) / len(bi_grams_list)}")
    print(f"Average tri-grams proportion: {sum(tri_grams_list) / len(tri_grams_list)}")
    print("--------------------------------------------------\n")


def main():
    raw_datasets = load_dataset(
                "paniniDot/sci_lay",
                "all",
            )

    full_plain_text = []
    full_technical_text = []
    full_article_text = []
    for split in ["train", "validation", "test"]:
        dataset = raw_datasets[split]

        full_article_text.extend(dataset["full_text"])
        full_plain_text.extend(dataset["plain_text"])
        full_technical_text.extend(dataset["technical_text"])

    print("Plain text:")
    compute_abstractiveness(full_article_text, full_plain_text)
    print("Technical text:")
    compute_abstractiveness(full_article_text, full_technical_text)


if __name__ == "__main__":
    main()