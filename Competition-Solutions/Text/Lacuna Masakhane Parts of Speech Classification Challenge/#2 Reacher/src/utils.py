import random, os
import numpy as np
import torch
import ast
import pandas as pd
import glob
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.optim import AdamW

    
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def convert_to_list(s):
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return None

def optimizer_scheduler(model,steps, NUM_EPOCHS, GRAD_ACCUM_STEPS, LR):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=LR)
        all_steps = NUM_EPOCHS*steps//GRAD_ACCUM_STEPS
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0.04*all_steps,
            num_training_steps=all_steps
        )
        return opt, sch

def preprocess_data():
    data_list = []
    for file_path in glob.glob('masakhane-pos/data/*/*.txt'):
        with open(file_path, 'r') as file:
                lines = file.readlines()
                targets = []
                texts = []
                langs = []
                for idx, line in enumerate(lines):
                    if idx==0:
                        sentence = ""
                        target = []
                    if line=='\n':
                        texts.append(sentence)
                        targets.append(target)
                        langs.append(file_path.split('/')[2])
                        sentence = ""
                        target = []
                        continue
                    text, target_ = line.split(' ')
                    sentence += ' ' + text
                    target.append(target_.strip())
                
                data_list.append(pd.DataFrame({'text': texts, 'target': targets, 'lang': langs}))
    train = pd.concat(data_list).reset_index(drop=True)
    train.to_csv('data/train.csv', index=False)
    return train

def preprocess_test(test):
    test["sen"] = test.Id.apply(lambda x: x.split('_')[0])
    sentences=[]
    for id_  in test.sen.unique():
        tmp = test[test.sen==id_].reset_index()
        sentence=""
        for i in range(len(tmp)):
            sentence+=tmp.iloc[i].Word +  " "
        sentences.append(sentence)
    new_test = pd.DataFrame({"text": sentences})
    new_test.to_csv('data/new_test.csv', index=False)
    return new_test

def create_luo_data():
    data_list = []
    with open('lacuna_pos_ner/language_corpus/luo/luo.txt', 'r') as file:
            lines = file.readlines()
    df = pd.DataFrame({"text": lines})
    df["text"] = df["text"].apply(lambda x: x.replace("\n", "").replace(".", " ."))
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def create_tsn_data():
    data_list = []

    with open('lacuna_pos_ner/language_corpus/tsn/tsn.txt', 'r') as file:
        lines = file.readlines()
    df = pd.DataFrame({"text": lines})
    df["text"] = df["text"].apply(lambda x: x.replace("\n", "").replace(".", " ."))
    df = df.drop_duplicates().reset_index(drop=True)
    return df
    
