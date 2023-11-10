import torch
from sklearn.metrics import accuracy_score
import numpy as np
from .dataset import *
from transformers import DataCollatorWithPadding, AutoTokenizer 
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from tqdm.auto import tqdm

def validate(loader, all_preds):
    preds = []
    y_test = []
    tr_accuracy = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bar=tqdm(enumerate(loader),total=len(loader))
    for idx, batch in bar:
                
        # MOVE BATCH TO GPU AND INFER
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch['labels'].to(device, dtype = torch.long)
    
        # INTERATE THROUGH EACH TEXT AND GET PRED
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = torch.tensor(all_preds[idx]).to('cuda').view(-1, 17) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    if np.random.rand()<0.3:
        print("Validation ACC: ", tr_accuracy/len(loader))
    
    
    return tr_accuracy/len(loader)

def get_loader(train, PRETRAINED_MODEL,labels_to_ids, FOLD, exp):
    test_dataset = train[train.fold.isin([FOLD])].reset_index(drop=True)
    all_preds = np.load(f"models/{exp}/OOF_fold{FOLD}.npy")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL) 
    testing_set = CustomDataset(test_dataset, tokenizer, labels_to_ids, 200, False)
    # TRAIN DATASET AND VALID DATASET

    test_params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 4,
                    'pin_memory':True,
                'collate_fn':DataCollatorWithPadding(tokenizer=tokenizer)
                    }
    testing_loader = DataLoader(testing_set, **test_params)
    return testing_loader, all_preds

def find_best_thresh(FOLD, train, PRETRAINED_MODEL,labels_to_ids, exp):
    testing_loader, all_preds = get_loader(train, PRETRAINED_MODEL,labels_to_ids, FOLD, exp)
    n = 17
    def f(x):
        # Choose the competition metric
        
        pred1 = all_preds.copy()*x
        
        
        score = validate(testing_loader, pred1)
        return 1 - score
    options = {'maxiter': 130}
    result = minimize(lambda x: f(x), [1]*n, method="Nelder-Mead", options=options)

    w = result.x
    return w

def postprocess(test, sub):
    test["Word"] = test.Word.str.lower()
    words1 = test[test.Language=='luo'].Word.unique()
    words2 = test[test.Language=='tsn'].Word.unique()
    for word in words1:
        counts = sub[test.Language=='luo'][test.Word==word].Pos.value_counts().values
        if len(counts)>1 and (counts[0]/counts[1])>1.2 and np.sum(counts)>=140:
            sub.loc[(test.Language=='luo') & (test.Word==word), "Pos"] = sub[test.Language=='luo'][test.Word==word].Pos.value_counts().index[0]
    for word in words2:
        counts = sub[test.Language=='tsn'][test.Word==word].Pos.value_counts().values
        if len(counts)>1 and (counts[0]/counts[1])>1.2 and np.sum(counts)>=140:
            sub.loc[(test.Language=='tsn') & (test.Word==word), "Pos"] = sub[test.Language=='tsn'][test.Word==word].Pos.value_counts().index[0]
    return sub