import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BertAdapterModel,
    AutoTokenizer,
    BertForMaskedLM,
)

from utils import *


def preprocess_data(dataset_path, balance_dataset):
    df = pd.read_csv(dataset_path)
    gender_specific_sentences = df[df.label == "gender-specific"].sentence
    gender_neutral_sentences = df[df.label == "gender-neutral"].sentence
    if balance_dataset:
        balanced_size = np.min(
            [gender_specific_sentences.shape[0], gender_neutral_sentences.shape[0]]
        )
        return (
            gender_specific_sentences[-balanced_size:],
            gender_neutral_sentences[-balanced_size:],
        )
    return gender_specific_sentences, gender_neutral_sentences


def calculate_mlm_score(
    sentences,
    model,
    model_name,
    device,
    tokenizer,
    func,
    feminine_tokens,
    masculine_tokens,
    normalize,
):
    with torch.no_grad():
        diffs = []
        for sent in tqdm(sentences):
            if (not isinstance(sent, str)) or sent.count("[MASK]") != 1:
                continue
            sent = sent.replace("[MASK]", tokenizer.mask_token)
            input_ids = tokenizer(
                PADDING_TEXT + sent if "xlnet" in model_name else sent,
                return_tensors="pt",
            )["input_ids"].to(device)
            mask_index = torch.where(input_ids[0] == tokenizer.mask_token_id)
            if "xlnet" in model_name:
                target_mapping = torch.zeros(
                    (1, 1, input_ids.shape[1]), dtype=torch.float
                ).to(device)
                target_mapping[0, 0, mask_index] = 1.0
                perm_mask = torch.zeros(
                    (1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float
                ).to(device)
                perm_mask[0, :, mask_index] = 1.0
            output = (
                model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
                if "xlnet" in model_name
                else model(input_ids)
            )
            target_word = (
                output[0][:, -1, :]
                if "xlnet" in model_name
                else output.logits[0, mask_index, :]
            )
            target_word_probs = (
                F.softmax(target_word, dim=-1)
                if target_word.sum(-1).item() != 1
                else target_word
            )

            if normalize:
                mask_index = np.zeros(target_word_probs.shape[-1], dtype=bool)
                mask_index[feminine_tokens + masculine_tokens] = True
                target_word_probs[:, ~mask_index] = 0
                target_word_probs = target_word_probs / target_word_probs.sum(-1)

            if func == torch.max:
                feminine_val = (
                    func(target_word_probs[:, feminine_tokens], dim=-1)
                    .values[0]
                    .cpu()
                    .numpy()
                )
                masculine_val = (
                    func(target_word_probs[:, masculine_tokens], dim=-1)
                    .values[0]
                    .cpu()
                    .numpy()
                )
            else:
                feminine_val = func(
                    target_word_probs[:, feminine_tokens], dim=-1
                ).item()
                masculine_val = func(
                    target_word_probs[:, masculine_tokens], dim=-1
                ).item()

            diffs.append(np.abs(masculine_val - feminine_val))
        return np.mean(diffs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="data/difair.csv",
        type=str,
        help="Path to the input .csv dataset",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="Name of the Hugging Face pretrained model",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="Path to the Hugging Face pretrained model",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        type=str,
        help="Path to the pretrained adapter modules",
    )
    parser.add_argument(
        "--masculine_words_path",
        default="data/masculine_words.txt",
        type=str,
        help="Path to text file containing masculine words",
    )
    parser.add_argument(
        "--feminine_words_path",
        default="data/feminine_words.txt",
        type=str,
        help="Path to text file containing feminine words",
    )
    parser.add_argument(
        "--no_balance_dataset",
        dest="balance_dataset",
        action="store_false",
        help="Toggle to balance/unbalance the dataset for tests",
    )
    parser.add_argument(
        "--no_normalization",
        dest="normalize",
        action="store_false",
        type=bool,
        help="Toggle to normalize/unnormalize probability distribution",
    )

    parser.set_defaults(balance_dataset=True)
    parser.set_defaults(normalize=False)

    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = args.model_name

    if args.adapter is not None:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        adapter_name = model.load_adapter(args.adapter, config="pfeiffer")
        model.set_active_adapters(adapter_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    elif "zari-bert" in args.model_name:
        model = BertForMaskedLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased" if "cda" in args.model_name else "bert-large-cased",
            use_fast=True,
        )
    elif "xlnet" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    print(f"evalating model {args.model_name} from {args.model_path}:")

    aggregation_func = torch.max
    feminine_words = read_words_file(args.feminine_words_path)
    masculine_words = read_words_file(args.masculine_words_path)
    feminine_tokens = list(set(words_to_tokens(feminine_words, tokenizer)))
    masculine_tokens = list(set(words_to_tokens(masculine_words, tokenizer)))

    gender_specific_sentences, gender_neutral_sentences = preprocess_data(
        args.dataset_path, args.balance_dataset
    )
    gender_specific_score = calculate_mlm_score(
        gender_specific_sentences,
        model,
        args.model_name,
        device,
        tokenizer,
        aggregation_func,
        feminine_tokens,
        masculine_tokens,
        args.normalize,
    )
    gender_neutral_score = 1 - calculate_mlm_score(
        gender_neutral_sentences,
        model,
        args.model_name,
        device,
        tokenizer,
        aggregation_func,
        feminine_tokens,
        masculine_tokens,
        args.normalize,
    )
    gender_bias_score = harmonic_mean(gender_specific_score, gender_neutral_score)

    print(
        f"model={args.adapter if args.adapter else args.model_path}, gender_specific_score={gender_specific_score}, gender_neutral_score={gender_neutral_score}, bias_score={gender_bias_score}"
    )


if __name__ == "__main__":
    main()
