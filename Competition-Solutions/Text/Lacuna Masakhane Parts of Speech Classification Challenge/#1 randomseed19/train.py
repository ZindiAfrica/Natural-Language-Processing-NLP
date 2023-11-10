import argparse
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

# from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    NllbTokenizer,
    NllbTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from masakhane_pos.m2m_100_encoder.modeling_m2m_100 import (
    M2M100ForTokenClassification,
)
from masakhane_pos.utils import (
    ID2LABEL,
    LABEL2ID,
    NLLB_LANG_MAP,
    PredictAndSaveCallback,
    compute_metrics,
    load_test_data,
    load_train_data,
    make_dataset,
    make_dataset_pseudo,
)

# ignore seqeval warnings for custom tags
warnings.filterwarnings("ignore", module="seqeval")

# languages are sorted by their public score in descending order
# when the model is trained using only that language
top_langs = [
    "lug",
    "ibo",
    "mos",
    "twi",
    "wol",
    "yor",
    "sna",
    "fon",
    "ewe",
    "nya",
    "kin",
    "zul",
    "xho",
    "swa",
    "hau",
    "bam",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_folder",
        default="masakhane-pos/data",
        help="Path to train data",
    )
    parser.add_argument(
        "--test_folder",
        default="zindi_masakhane_pos/data",
        help="Path to train data",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        required=True,
        help=(
            "Languages to train on, splitted by space, 'all' for all languages"
        ),
    )
    parser.add_argument(
        "-p",
        "--pseudo_labels",
        type=str,
        default="",
        required=False,
        help=(
            "csv file with predictions for luo and tsn to use as pseudo labels"
        ),
    )
    parser.add_argument(
        "--wandb_project",
        default="masakhane_pos",
        help="Name for wandb project if using wandb",
    )
    parser.add_argument("--lora", action="store_true", help="Train LoRA")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Total batch size, must be multiple of 16, device_batch_size"
            " hardcoded to 16 for Google Colab"
        ),
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Ratio for learning rate warmup",
    )

    args = parser.parse_args()
    # train on the given languages, validate on all other presented in the dataset
    languages_train = args.languages[0].split(" ")

    # set parameters
    DATA_FOLDER_TEST = Path(args.test_folder)
    DATA_FOLDER_TRAIN = Path(args.train_folder)

    LORA = args.lora
    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    REPORT_TO = "tensorboard"  # tensorboard wandb
    RANDOM_SEED = 19
    LEARNING_RATE = args.lr  # 3e-4
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER_TYPE = "constant_with_warmup"  # linear constant_with_warmup
    BATCH_SIZE = args.batch_size  # 64
    DEVICE_BATCH_SIZE = 16
    assert BATCH_SIZE % DEVICE_BATCH_SIZE == 0
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // DEVICE_BATCH_SIZE
    NUM_TRAIN_EPOCHS = args.num_train_epochs  # 10
    WARMUP_RATIO = args.warmup_ratio  # 0.05
    SAVE_TOTAL_LIMIT = None
    OPTIM = "adamw_torch"
    LOGGING_STEPS = 5
    METRIC_FOR_BEST_MODEL = "accuracy"
    GREATER_IS_BETTER = True
    FP16 = True
    GRADIENT_CHECKPOINTING = False

    RUN_NAME = (
        f"{'_'.join(i for i in languages_train)}_{MODEL_NAME.split('/')[-1]}_{'lora' if LORA else 'full'}_{'pseudo' if args.pseudo_labels else ''}_{NUM_TRAIN_EPOCHS}ep_lr{LEARNING_RATE}_{LR_SCHEDULER_TYPE}_bs{BATCH_SIZE}"
    )
    PREDICTIONS_FOLDER = f"pred_{RUN_NAME}"
    LOGGING_DIR = f"logs_{RUN_NAME}"
    if REPORT_TO == "wandb":
        # load_dotenv()
        # WANDB_KEY = os.getenv("WANDB_KEY")
        WANDB_KEY = input("Enter wandb key: ")
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project=args.wandb_project,
            name=f"{'_'.join(i for i in languages_train)}",
        )

    set_seed(RANDOM_SEED)

    # load data
    df_train = load_train_data(DATA_FOLDER_TRAIN)
    df_test = load_test_data(DATA_FOLDER_TEST / "Test.csv")
    df_sub = pd.read_csv(DATA_FOLDER_TEST / "SampleSubmission.csv")

    # load tokenizer and prepare dataset
    tokenizer = NllbTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = make_dataset(df_train, df_test, tokenizer)
    if languages_train[0] != "all":
        # train on given languages
        dataset["train"] = dataset["train"].filter(
            lambda x: x["lang"] in languages_train
        )
        # validate on the remaining languages
        dataset["dev"] = dataset["dev"].filter(
            lambda x: x["lang"]
            in [i for i in top_langs if i not in languages_train]
        )

    # add pseudo labels
    if args.pseudo_labels:
        dataset_pseudo = make_dataset_pseudo(
            args.pseudo_labels, df_test, tokenizer
        )
        dataset["train"] = concatenate_datasets(
            [dataset["train"], dataset_pseudo]
        )

    # data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # prepare model
    model = M2M100ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # prepare LoRA
    if LORA:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
            ],
            lora_dropout=0.05,
            bias="all",
            task_type=TaskType.TOKEN_CLS,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # set training arguments
    training_args = TrainingArguments(
        output_dir=RUN_NAME,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        per_device_train_batch_size=DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=REPORT_TO,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim=OPTIM,
        logging_steps=LOGGING_STEPS,
        logging_dir=LOGGING_DIR,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=FP16,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        seed=RANDOM_SEED,
        run_name=RUN_NAME,
    )

    # define trainer, callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(
        PredictAndSaveCallback(
            trainer, dataset["test"], df_sub, RUN_NAME, PREDICTIONS_FOLDER
        ),
    )

    # train
    trainer.train()

    if REPORT_TO == "wandb":
        wandb.finish()
