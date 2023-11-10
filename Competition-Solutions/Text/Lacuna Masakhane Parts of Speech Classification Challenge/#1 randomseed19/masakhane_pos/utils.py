from __future__ import annotations

from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import TrainerCallback, set_seed

set_seed(19)


# constants
LABEL2ID = {
    "NOUN": 0,
    "VERB": 1,
    "PUNCT": 2,
    "PROPN": 3,
    "ADP": 4,
    "PRON": 5,
    "DET": 6,
    "ADJ": 7,
    "AUX": 8,
    "ADV": 9,
    "CCONJ": 10,
    "PART": 11,
    "SCONJ": 12,
    "NUM": 13,
    "X": 14,
    "SYM": 15,
    "INTJ": 16,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

NLLB_LANG_MAP = {
    # train
    "bam": "bam_Latn",
    "bbj": None,  # not in the NLLB
    "ewe": "ewe_Latn",
    "fon": "fon_Latn",
    "hau": "hau_Latn",
    "ibo": "ibo_Latn",
    "kin": "kin_Latn",
    "lug": "lug_Latn",
    "mos": "mos_Latn",
    "nya": "nya_Latn",
    "pcm": None,  # not in the NLLB
    "sna": "sna_Latn",
    "swa": "swh_Latn",  # swh instead of swa
    "twi": "twi_Latn",
    "wol": "wol_Latn",
    "xho": "xho_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
    # test
    "luo": "luo_Latn",
    "tsn": "tsn_Latn",
}


class ParseError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def load_pos_file(file_path: Path) -> list[dict]:
    """Loads txt file with POS labels to list of dicts"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        sentence_id = 0
        word_id = 0
        for line in f:
            line = line.strip()
            if not line:  # empty line indicates the end of a sentence
                sentence_id += 1
                word_id = 0
            else:
                parts = line.rsplit(" ")
                # ensure there are at least two columns (word and tag)
                try:
                    word, pos = parts[0], parts[1]
                    data.append(
                        {
                            "sentence_id": sentence_id,
                            "word_id": word_id,
                            "word": word,
                            "pos": pos,
                        }
                    )
                    word_id += 1
                except:
                    raise ParseError(f"Cannot parse line {line}")
    return data


def load_train_data(input_folder: Path) -> pd.DataFrame:
    """Loads all train data into pandas DataFrame"""
    train_data = []
    for folder in input_folder.glob("*"):
        if folder.is_dir():
            lang = folder.name
            for filepath in folder.glob("*.txt"):
                split = filepath.name.rsplit(".", 1)[0]
                pos_data = load_pos_file(filepath)
                for d in pos_data:
                    d["lang"] = lang
                    d["split"] = split
                    train_data.append(d)
    df_train = pd.DataFrame.from_records(train_data)
    columns = ["lang", "split", "sentence_id", "word_id", "word", "pos"]
    df_train = df_train[columns]
    return df_train


def load_test_data(input_file: Path) -> pd.DataFrame:
    """Loads test data into pandas DataFrame"""
    df_test = pd.read_csv(input_file)
    df_test.columns = ["id", "word", "lang", "pos"]  # make same names
    df_test["split"] = "test"
    df_test["sentence_id"] = df_test["id"].apply(lambda x: x.split("_")[0])
    df_test["word_id"] = df_test["id"].apply(lambda x: x.split("_")[1])
    # without pos column
    columns = ["lang", "split", "sentence_id", "word_id", "word"]
    df_test = df_test[columns]
    return df_test


def tokenize_inputs(example, tokenizer):
    """Prepare inputs for model
    processed only one example, doesn't work with batching"""
    tokenizer.src_lang = NLLB_LANG_MAP.get(example["lang"])
    tokenized_inputs = tokenizer(
        example["word"], is_split_into_words=True, add_special_tokens=True
    )
    example["word_ids"] = tokenized_inputs.word_ids()
    example["input_ids"] = tokenized_inputs["input_ids"]
    example["attention_mask"] = tokenized_inputs["attention_mask"]
    return example


def align_labels(example):
    """Aligns labels with input tokens.
    Labels only first token of multitoken words."""
    label_ids = []
    previous_word_idx = None
    for word_idx in example["word_ids"]:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # label only the first token of a given word
            label_ids.append(LABEL2ID[example["pos"][word_idx]])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    example["labels"] = label_ids
    return example


def make_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame, tokenizer):
    """Main function that prepares dataset"""
    # Combines initial test of masakhane_pos with train. Will validate on dev
    # so we have 2 splits: train (initial train + initial test) and dev
    # new test will be lua and tsn languages not presented in masakhane-pos dataset
    df_train.loc[df_train["split"] == "test", "split"] = "train"

    # add constant label to test for easier processing predictions later
    df_test["pos"] = "X"

    df_train = df_train.groupby(["lang", "split", "sentence_id"]).agg(
        {"word": list, "pos": list}
    )
    df_test = df_test.groupby(["lang", "split", "sentence_id"]).agg(
        {"word": list, "pos": list}
    )

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                df_train.loc[
                    df_train.index.get_level_values("split") == "train"
                ]
            ),
            "dev": Dataset.from_pandas(
                df_train.loc[df_train.index.get_level_values("split") == "dev"]
            ),
            "test": Dataset.from_pandas(df_test),
        }
    )
    # keep only languages that present in NLLB
    print("Filtering NLLB languages")
    dataset = dataset.filter(
        lambda example: example["lang"]
        in (k for k, v in NLLB_LANG_MAP.items() if v)
    )

    # tokenize
    print("Tokenizing")
    dataset = dataset.map(
        tokenize_inputs,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # align labels
    print("Aligning labels")
    dataset = dataset.map(
        align_labels,
        batched=False,
    )

    # remove excess columns
    columns_to_remove = [
        "split",
        # "lang",
        "word",
        "pos",
    ]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def make_dataset_pseudo(test_pred_file, df_test, tokenizer):
    """Makes dataset with luo and tsn data labeled by model trained on masakhane_pos data"""
    df_test_pred = pd.read_csv(test_pred_file)
    df_pseudo = df_test.copy()
    df_pseudo["pos"] = df_test_pred["Pos"]
    # cast sentence_id to int as in train dataset
    df_pseudo["sentence_id"] = pd.factorize(df_pseudo["sentence_id"])[0]
    df_pseudo = df_pseudo.groupby(["lang", "split", "sentence_id"]).agg(
        {"word": list, "pos": list}
    )
    dataset_pseudo = Dataset.from_pandas(df_pseudo)
    dataset_pseudo = dataset_pseudo.map(
        tokenize_inputs,
        batched=False,
        fn_kwargs={"tokenizer": tokenizer},
    )
    dataset_pseudo = dataset_pseudo.map(
        align_labels,
        batched=False,
    )
    columns_to_remove = [
        "split",
        # "lang",
        "word",
        "pos",
    ]
    dataset_pseudo = dataset_pseudo.remove_columns(columns_to_remove)
    return dataset_pseudo


# Metrics
metric = evaluate.load("seqeval")


def compute_metrics(p: tuple):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class PredictAndSaveCallback(TrainerCallback):
    """Predicts labels for test dataset and saves them to csv on each validation step"""

    def __init__(self, trainer, test_dataset, df_sub, run_name, out_dir):
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.df_sub = df_sub
        self.run_name = run_name
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir()

    def on_evaluate(self, args, state, control, **kwargs):
        test_preds = self.trainer.predict(self.test_dataset)
        predictions, labels = (test_preds.predictions, test_preds.label_ids)
        predictions = np.argmax(predictions, axis=2)

        # remove ignored index (special tokens)
        test_predictions = [
            [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # make id to prediction mapping in case of shuffle
        id2pred = {}
        for sent_id, sent_pred in zip(
            self.test_dataset["sentence_id"], test_predictions
        ):
            for i, pred_pos_i in enumerate(sent_pred):
                id2pred[f"{sent_id}_{i}"] = pred_pos_i
        self.df_sub["Pos"] = self.df_sub["Id"].map(id2pred)
        self.df_sub.to_csv(
            self.out_dir / f"xx_{self.run_name}_{state.global_step}.csv",
            index=False,
        )
