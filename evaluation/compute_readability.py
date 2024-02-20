import argparse
import textstat
from tqdm import tqdm
from readability import Readability
from datasets import load_dataset

METRICS = ["fkgl", "cli", "gf", "dcrs"]


def print_read(full_text):
    r = Readability(full_text)

    output = "------------------------\n"
    output += f"Flesch-Kincaid grade level: {r.flesch_kincaid().score:.2f}\n"
    output += f"Gunning fog index: {r.gunning_fog().score:.2f}\n"
    output += f"Coleman-Liau index: {r.coleman_liau().score:.2f}\n"
    output += f"Dale-Chall readability score: {r.dale_chall().score:.2f}\n"
    output += f"Average readability metrics score: {((r.flesch_kincaid().score + r.gunning_fog().score + r.coleman_liau().score + r.dale_chall().score) / 4):.2f}\n"
    output += "------------------------"
    print(output)


def main():
    raw_datasets = load_dataset(
                "paniniDot/sci_lay",
                "all",
            )

    full_plain_text = ""
    full_technical_text = ""
    for split in ["test"]:
        dataset = raw_datasets[split]

        full_plain_text += " ".join(dataset["plain_text"])
        full_technical_text += " ".join(dataset["technical_text"])

    print("Plain text:")
    print_read(full_plain_text)
    print("Technical text:")
    print_read(full_technical_text)


if __name__ == "__main__":
    main()

