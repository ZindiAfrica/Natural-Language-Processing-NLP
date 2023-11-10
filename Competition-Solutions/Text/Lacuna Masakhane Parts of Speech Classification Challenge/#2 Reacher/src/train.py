from .dataset import CustomDataset
from .model import TransformerModel
from .utils import optimizer_scheduler
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,

)
import torch
from torch import nn
import numpy as np
import gc

def train_epoch(model, training_loader, optimizer,scaler, epoch,GRAD_ACCUM_STEPS = 2, USE_AMP=True, scheduler=None, label_smooth=0.1, grad_norm=20):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    criterion=nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smooth)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # put model in training mode
    model.train()
    bar=tqdm(enumerate(training_loader),total=len(training_loader))
    steps = len(training_loader)
    for idx, batch in bar:


        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)
        
        # cutmix
        if np.random.uniform()<0.2:
            cut=0.15
            perm=torch.randperm(ids.shape[0]).cuda()
            rand_len=int(ids.shape[1]*cut)
            start=np.random.randint(ids.shape[1]-int(ids.shape[1]*cut))
            ids[:,start:start+rand_len]=ids[perm,start:start+rand_len]
            mask[:,start:start+rand_len]=mask[perm,start:start+rand_len]
            labels[:,start:start+rand_len]=labels[perm,start:start+rand_len]


        with torch.cuda.amp.autocast(enabled=USE_AMP):
            output = model(ids,mask)

            output=output.reshape(-1,17)
            labels=labels.reshape(-1)

            loss_mask=labels!=-100
            labels[labels==-100]=0

            loss=criterion(output,labels)
            loss=torch.masked_select(loss,loss_mask).mean()

        tr_loss += loss.item()

        bar.set_postfix({'train_loss': tr_loss/(idx+1)})
        scaler.scale(loss).backward()
        nb_tr_steps += 1
        if (idx + 1) % GRAD_ACCUM_STEPS == 0 or idx == steps:
            if grad_norm!=0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = 0
    print(f"Training loss epoch: {epoch_loss}")
    return epoch_loss, tr_accuracy
def validate(model, loader, USE_AMP=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    preds = []
    y_test = []
    tr_accuracy = 0
    bar=tqdm(enumerate(loader),total=len(loader))
    for idx, batch in bar:
                
        # MOVE BATCH TO GPU AND INFER
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch['labels'].to(device, dtype = torch.long)
        
        with torch.autocast(device_type="cuda"):
            output = model(ids,mask)
        all_preds = output
        preds.append(torch.nn.functional.softmax(output, dim=-1).detach().cpu().numpy())
        y_test.append(labels.detach().cpu().numpy())
    
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = all_preds.view(-1, 17) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
            

    
        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    print("Validation ACC: ", tr_accuracy/len(loader))
    preds = np.concatenate(preds)

    return tr_accuracy/len(loader), preds

def train_folds(train, exp,labels_to_ids, PRETRAINED_MODEL,LRs, BATCH_SIZE=4, LR=1e-5, USE_AMP=True, MAX_SEQ_LENGTH=200, NUM_EPOCHS=1, GRAD_ACCUM_STEPS=2, label_smooth=0.1, grad_norm=20, luo=None, tsn=None, luo_n=3000, tsn_n=3000):

    for FOLD in range(5):
        print('-------------------------------------------------------------')
        print(f'Fold {FOLD}')
        print('-------------------------------------------------------------')
        train_dataset = train[~train.fold.isin([FOLD])].reset_index(drop=True)
        test_dataset = train[train.fold.isin([FOLD])].reset_index(drop=True)
        
        if luo is not None:
            train_dataset = pd.concat(
                [train_dataset, luo.sample(luo_n), tsn.sample(tsn_n)]
            ).sample(frac=1).reset_index()
        
        print("FULL Dataset: {}".format(train.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))
        
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL) 
        training_set = CustomDataset(train_dataset, tokenizer,labels_to_ids, MAX_SEQ_LENGTH, False, True)
        testing_set = CustomDataset(test_dataset, tokenizer, labels_to_ids, MAX_SEQ_LENGTH, False)
        # TRAIN DATASET AND VALID DATASET
        train_params = {'batch_size': BATCH_SIZE,
                        'shuffle': True,
                        'num_workers': 4,
                        'pin_memory':True,
                        'collate_fn':DataCollatorWithPadding(tokenizer=tokenizer)
                        }
        
        test_params = {'batch_size': 4,
                        'shuffle': False,
                        'num_workers': 4,
                        'pin_memory':True,
                       'collate_fn':DataCollatorWithPadding(tokenizer=tokenizer)
                        }
        
        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        
        model = TransformerModel(PRETRAINED_MODEL).to('cuda')
        
        optimizer, _ = optimizer_scheduler(model,len(training_loader),NUM_EPOCHS, GRAD_ACCUM_STEPS, LR)
        scheduler=None
        
        best_score=0
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        for epoch in range(NUM_EPOCHS):
        
                print(f"### Training epoch: {epoch + 1}")
                for g in optimizer.param_groups: 
                    g['lr'] = LRs[epoch]
                lr = optimizer.param_groups[0]['lr']
                print(f'### LR = {lr}\n')

                train_epoch(model, training_loader, optimizer,scaler, epoch, GRAD_ACCUM_STEPS=GRAD_ACCUM_STEPS, label_smooth=label_smooth, grad_norm=grad_norm)
                print(f"### Evaluating epoch: {epoch + 1}")
                score, preds = validate(model, testing_loader, )
                torch.cuda.empty_cache()
                gc.collect()
                if score> best_score:
                    best_score = score
                    torch.save(model.state_dict(), f'models/{exp}/fold{FOLD}.pt')
                    np.save(f'models/{exp}/OOF_fold{FOLD}.npy', preds)
    
    