#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Import...')

import os
import gc
import re
import sys
sys.path.append('../../lib')
import glob
import math
import json
from datetime import datetime as dt
import time
import pickle
import random
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd

import torch
print('torch:', torch.__version__)
import transformers as tr
print('tr:', tr.__version__)

from vecxoz_utils import ArgumentParserExtended
from vecxoz_utils import seeder
from vecxoz_utils import prepare_trim
from vecxoz_utils import shuffle_batches
from vecxoz_utils import create_cv_split
from vecxoz_utils import keep_last_ckpt_torch

from datasets import load_metric
wer_score = load_metric('wer')

from rouge import Rouge
rouge_score = Rouge()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Parse CMD...')

parser = ArgumentParserExtended()

parser.add_str('--model_dir_or_name', default='google/mt5-base', help='Model name or directory containing weights and vocab')
parser.add_int('--max_len',           default=96,            help='Maximum length of input sequence (training phase)')
parser.add_int('--max_len_out',       default=20,            help='Maximum length of output sequence (training phase)')
parser.add_int('--n_beams',           default=1,             help='Number of beams (training phase)')
parser.add_int('--max_len_out_infer', default=96,            help='Maximum length of output sequence (inference phase)')
parser.add_int('--n_beams_infer',     default=4,             help='Number of beams (inference phase)')
parser.add_int('--seed',              default=733796,        help='Random seed global')
parser.add_int('--seed_for_data',     default=33,            help='Random seed for data split')
parser.add_str('--device',            default='cuda:0',      help='Device')
parser.add_int('--print_n_batches',   default=100,           help='Report interval')
parser.add_bool('--torch_mixed_precision', default=False,    help='Whether to use torch.cuda.amp')
parser.add_int('--n_folds',           default=10,            help='Number of folds')
parser.add_int('--initial_fold',      default=0,             help='Initial fold (from 0)')
parser.add_int('--final_fold',        default=1,             help='Final fold (from 1). None means all folds i.e. equal to n_folds')
parser.add_str('--lib_dir',           default='../../lib',   help='Import directory')
parser.add_int('--n_steps_train',     default=None,          help='Number of steps to train per epoch. None means all examples')
parser.add_str('--torch_monitor',     default='val_rouge',   help='Value to monitor during training to apply checkpoint callback, etc.')
parser.add_str('--torch_scheduler_mode', default='max',      help='Scheduler model (min or max) depending on monitored metric')
parser.add_bool('--skip_training',    default=False,         help='Whether to skip training for initial fold')
parser.add_bool('--use_lr_schedule',  default=False,         help='Whether to use LR scheduler')

parser.add_str('--data_dir',          default='../../data',  help='Data directory containig CSV files')
parser.add_str('--data_preds_dir',    default='preds',       help='Directory to save predictions')
parser.add_str('--out_dir',           default='./',          help='Directory to save model checkpoints')
parser.add_str('--job',               default='train_val_test_score_subm', help='Job to perform. A combination of words: train, val, test, score, subm. E.g. train_val_test')
parser.add_int('--n_examples_train_total',  default=22353,   help='Number of training examples. This value is used to define an epoch')
parser.add_int('--n_epochs',          default=6,             help='Number of epochs to train')
parser.add_int('--batch_size',        default=4,             help='Batch size')
parser.add_int('--accumulation_steps',default=8,             help='Accumulation steps')
parser.add_int('--batch_size_infer',  default=32,            help='Batch size for inference')
parser.add_float('--lr',              default=1e-4,          help='Learning rate')
parser.add_int('--aug_number',        default=0,             help='Number of train-time augmentations. 0 means no augmentation')
parser.add_int('--tta_number',        default=0,             help='Number of test-time augmentations. 0 means no augmentation. In result there will be (tta_number + 1) predictions')
parser.add_bool('--save_any',         default=True,          help='Whether to save checkpoint regardless of validation score')

args = parser.parse_args() # add args=[] to run in notebook

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Date-time
args.date_time=dt.now().strftime('%Y%m%d-%H%M%S-%f')

# Set seed
_ = seeder(seed=args.seed, general=True, te=False, to=True)

# Create dirs
if not os.path.exists(args.data_preds_dir):
    os.makedirs(args.data_preds_dir)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Number of sub-train examples i.e. all folds except one (e.g. 4/5 of full train)
args.n_examples_train = args.n_examples_train_total - (args.n_examples_train_total // args.n_folds)

# Training steps
if args.n_steps_train is None:
    args.n_steps_train = args.n_examples_train // args.batch_size

# Set import dirs
sys.path.append(args.lib_dir)

# Folds
assert args.initial_fold <= args.n_folds - 1, 'Incorrect initial_fold'
if args.final_fold is None:
    args.final_fold = args.n_folds
else:
    assert args.final_fold <= args.n_folds and args.final_fold > args.initial_fold, 'Incorrect final_fold'

print('Settings')
print(parser.args_repr(args, False))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def rouge_dot_safe(lines_pred, lines_true):
    """
    rouge-1, f-measure, mean

    Implementation of ROUGE metric in packge "rouge" gives an exception if
    line consists of dots only. In current function we check this and if such
    case is found we just replace it with 'a'
    """
    lines_pred_ok = []
    for line in lines_pred:
        if list(set(list(line))) == ['.']:
            lines_pred_ok.append('a')
        else:
            lines_pred_ok.append(line)
    #
    lines_true_ok = []
    for line in lines_true:
        if list(set(list(line))) == ['.']:
            lines_true_ok.append('a')
        else:
            lines_true_ok.append(line)
    #
    score = rouge_score.get_scores(lines_pred_ok, 
                                   lines_true_ok, 
                                   avg=True, 
                                   ignore_empty=True)['rouge-1']['f']
    return score

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Load CSV and create CV split...')

train_df, test_df = create_cv_split(os.path.join(args.data_dir, 'Train_Ewe.csv'), 
                                    os.path.join(args.data_dir, 'Test_Ewe.csv'), 
                                    col_label='Target', 
                                    col_group=None, 
                                    n_folds=args.n_folds, 
                                    splitter='kf',
                                    random_state=args.seed_for_data)

print(train_df['French'].iloc[0]) # Grand P est malade de la progéria ...

print('Init tokenizer')
tokenizer = tr.AutoTokenizer.from_pretrained(args.model_dir_or_name)
print(tokenizer.__class__.__name__) # T5TokenizerFast

print('Encode')

encoded_dict_french_train = tokenizer.batch_encode_plus(
    list(train_df['French'].values),
    max_length=args.max_len,
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    return_token_type_ids=False,
    return_attention_mask=False,)

encoded_dict_target_train = tokenizer.batch_encode_plus(
    list(train_df['Target'].values),
    max_length=args.max_len,
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    return_token_type_ids=False,
    return_attention_mask=False,)

encoded_dict_french_test = tokenizer.batch_encode_plus(
    list(test_df['French'].values),
    max_length=args.max_len,
    add_special_tokens=True,
    truncation=True,
    padding='max_length',
    return_token_type_ids=False,
    return_attention_mask=False,)

X = np.array(encoded_dict_french_train['input_ids'], dtype=np.int32)
Y = np.array(encoded_dict_target_train['input_ids'], dtype=np.int32)
X_test = np.array(encoded_dict_french_test['input_ids'], dtype=np.int32)
fold_ids = train_df['fold_id'].values.astype(np.int32)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Enter training loop...')

if args.job != 'score' and args.job != 'subm' and args.job != 'score_subm':
    for fold_id in range(args.initial_fold, args.final_fold):
        print('\n*****')
        print('Fold:', fold_id)
        print('*****\n')
        #--------------------------------------------------------------------------
        best_score = 0
        start = None
        #--------------------------------------------------------------------------
        if args.torch_mixed_precision:
            print('Using AMP')
        else:
            print('Using default precision')
        # Create fresh instance for each fold
        scaler = torch.cuda.amp.GradScaler(enabled=args.torch_mixed_precision)
        #--------------------------------------------------------------------------
        print('FULL BATCH SHAPE: %d x %d' % (args.batch_size,
                                             args.max_len,))
        #--------------------------------------------------------------------------
        print('Init datasets')
        X_train = X[fold_ids != fold_id]
        X_val = X[fold_ids == fold_id]
        Y_train = Y[fold_ids != fold_id]
        Y_val = Y[fold_ids == fold_id]
        #--------------------------------------------------------------------------
        print('Init model')
        config = tr.MT5Config.from_pretrained(args.model_dir_or_name)
        model = tr.MT5ForConditionalGeneration(config)
        device = torch.device(args.device)
        model = model.to(device)
        if 'train' in args.job:
            pp = 'model-prefinetuned-ewe.bin'
            print('Loading pre-finetuned model:', pp)
            model.load_state_dict(torch.load(pp))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.torch_scheduler_mode, factor=0.5, patience=7, verbose=True)
        #-------------------------------------------------------------------------- 
        # TRAIN
        #--------------------------------------------------------------------------
        if 'train' in args.job and not args.skip_training:
            print('Fit (fold %d)' % fold_id)
            #
            for epoch in range(args.n_epochs):
                print('Epoch %d of %d' % (epoch, args.n_epochs))
                start = time.time()
                #
                #--------------------------------------------------------------------------
                # DATA INIT with group-trim (use this code block instead the one outside epoch loop)
                #--------------------------------------------------------------------------
                # TRAIN
                # Shuffle trimmed training data before each epoch
                # Stage-1: shuffle all examples to make result of "prepare_trim" different
                #          examples with the same zero count are indistinguishable for sorting algo so final sorted order will depend on initial order
                shuffle_ids = shuffle_batches(X_train, batch_size=1)
                X_train, Y_train, zero_counts_train, ids_back_train = prepare_trim(X_train[shuffle_ids], Y_train[shuffle_ids], pad_token_id=tokenizer.pad_token_id)
                # Stage-2: shuffle batches
                #          this shuffle will preserve same (close) zero count in each batch
                shuffle_ids = shuffle_batches(X_train, batch_size=args.batch_size)
                td_train = torch.utils.data.TensorDataset(torch.tensor(X_train[shuffle_ids], dtype=torch.long),
                                                          torch.tensor(Y_train[shuffle_ids], dtype=torch.long),
                                                          torch.tensor(zero_counts_train[shuffle_ids], dtype=torch.long))
                train_loader = torch.utils.data.DataLoader(td_train, batch_size=args.batch_size, shuffle=False)
                #
                # VAL
                X_val, Y_val, zero_counts_val, ids_back_val = prepare_trim(X_val, Y_val, pad_token_id=tokenizer.pad_token_id)
                td_val = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                                torch.tensor(Y_val, dtype=torch.long),
                                                torch.tensor(zero_counts_val, dtype=torch.long))
                val_loader = torch.utils.data.DataLoader(td_val, batch_size=args.batch_size_infer, shuffle=False)
                #
                avg_loss = 0.0
                avg_wer = 0.0
                avg_rouge = 0.0
                #
                model.train()
                optimizer.zero_grad()
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(train_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    y_batch = y_batch[:, :(args.max_len - min_zero_count)]
                    #
                    with torch.cuda.amp.autocast(enabled=args.torch_mixed_precision):
                        res = model(x_batch.to(device), 
                                    labels=y_batch.to(device), 
                                    attention_mask=(x_batch != tokenizer.pad_token_id).long().to(device))
                        loss = res.loss
                        logits = res.logits
                        loss = loss / args.accumulation_steps
                    scaler.scale(loss).backward()
                    if (i+1) % args.accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    # Metrics
                    avg_loss += loss.item() / len(train_loader)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    #
                    current_text_pred = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                    current_text_true = tokenizer.batch_decode(y_batch, skip_special_tokens=True) # group_tokens=False
                    #
                    try:
                        current_wer = wer_score.compute(predictions=current_text_pred, references=current_text_true)
                        current_rouge = rouge_dot_safe(current_text_pred, current_text_true)
                        avg_wer += current_wer / len(train_loader)
                        avg_rouge += current_rouge / len(train_loader)
                    except:
                        print('Exception in metric')
                    #
                    if not i % args.print_n_batches:
                        print('Batch: %6d    Loss: %.6f    ROUGE: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_rouge, avg_wer, (time.time() - start)))
                #
                #--------------------------------------------------------------------------
                # EVAL after each epoch
                #--------------------------------------------------------------------------
                print('Eval...')
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                avg_rouge = 0.0
                mean_loss = 0.0
                mean_wer = 0.0
                mean_rouge = 0.0
                losses = []
                wers = []
                rouges = []
                preds = []
                text_pred = []
                text_true = []
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(val_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    y_batch = y_batch[:, :(args.max_len - min_zero_count)]
                    with torch.no_grad():
                        #
                        predicted_ids = model.generate(x_batch.to(device), max_length=args.max_len_out, num_beams=args.n_beams)
                        #
                        current_text_pred = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                        current_text_true = tokenizer.batch_decode(y_batch, skip_special_tokens=True)
                        #
                        try:
                            current_wer = wer_score.compute(predictions=current_text_pred, references=current_text_true)
                            current_rouge = rouge_dot_safe(current_text_pred, current_text_true)
                            avg_wer += current_wer / len(val_loader)
                            avg_rouge += current_rouge / len(val_loader)
                        except:
                            print('Exception in metric')
                        #
                        # Save results for further processing
                        wers.append(current_wer)
                        rouges.append(current_rouge)
                        text_pred.append(current_text_pred)
                        text_true.append(current_text_true)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    ROUGE: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_rouge, avg_wer, (time.time() - start)))
                        #
                # Compute mean metrics for VAL set in each epoch
                mean_wer = np.mean(wers)
                mean_rouge = np.mean(rouges)
                # Save model if score is better
                print('MEAN val LOSS: %.6f    MEAN val ROUGE: %.6f    MEAN val WER: %.6f' % (mean_loss, mean_rouge, mean_wer))
                print('Epoch time: %6d sec' % (time.time() - start))
                if mean_rouge >= best_score or args.save_any:
                    print('Saving model (%.6f vs. %.6f)...' % (mean_rouge, best_score))
                    best_score = mean_rouge
                    p = os.path.join(args.out_dir, 'model-best-f%d-e%03d-%.4f.bin' % (fold_id, epoch, mean_rouge))
                    torch.save(model.state_dict(), p)
                    print('Saved model:', p)
                else:
                    print('Mean ROUGE is not better (%.6f vs. %.6f): NOT saving the model' % (mean_rouge, best_score))
                #
                # Remove all previous checkpoints
                keep_last_ckpt_torch(os.path.join(args.out_dir, 'model-best-f%d-e*.bin' % fold_id))
                #
                if args.use_lr_schedule:
                    if args.torch_monitor == 'val_rouge':
                        scheduler.step(mean_rouge)
                    elif args.torch_monitor == 'val_loss':
                        scheduler.step(loss.item())
        #--------------------------------------------------------------------------
        # PREDICT VAL and TEST after each fold
        #--------------------------------------------------------------------------
        # Load best model for fold
        m = sorted(glob.glob('model-best-f%d*.bin' % fold_id))[-1]
        print('Load model (fold %d): %s' % (fold_id, m))
        model.load_state_dict(torch.load(m))
        if not start:
            start = time.time()
        args.skip_training = False # skip only once (do not skip for all next folds)
        #--------------------------------------------------------------------------
        # TTA
        #--------------------------------------------------------------------------
        for tta_id in range(args.tta_number + 1):
            # Create VAL and TEST datasets with TTA transforms corresponding to AUG transforms seen during training
            print('Init datasets for prediction (fold %d, tta %d)' % (fold_id, tta_id))
            #
            # For inference we use regular data without grouping
            X_val = X[fold_ids == fold_id]
            Y_val = Y[fold_ids == fold_id]

            zero_counts_val = np.sum(X_val == tokenizer.pad_token_id, axis=1)
            zero_counts_test = np.sum(X_test == tokenizer.pad_token_id, axis=1)

            td_val = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long), 
                                                    torch.tensor(Y_val, dtype=torch.long), 
                                                    torch.tensor(zero_counts_val, dtype=torch.long),)
            td_test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long),  
                                                     torch.tensor(zero_counts_test, dtype=torch.long),)

            val_loader = torch.utils.data.DataLoader(td_val, batch_size=args.batch_size_infer, shuffle=False)
            test_loader = torch.utils.data.DataLoader(td_test, batch_size=args.batch_size_infer, shuffle=False)
            #--------------------------------------------------------------------------
            # Predict VAL
            #--------------------------------------------------------------------------
            if 'val' in args.job:
                print('Predict VAL (fold %d, tta %d)' % (fold_id, tta_id))
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                avg_rouge = 0.0
                mean_loss = 0.0
                mean_wer = 0.0
                mean_rouge = 0.0
                losses = []
                wers = []
                rouges = []
                preds = []
                text_pred = []
                text_true = []
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(val_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    y_batch = y_batch[:, :(args.max_len - min_zero_count)]
                    with torch.no_grad():
                        #
                        predicted_ids = model.generate(x_batch.to(device), max_length=args.max_len_out_infer, num_beams=args.n_beams_infer)
                        #
                        current_text_pred = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                        current_text_true = tokenizer.batch_decode(y_batch, skip_special_tokens=True) # group_tokens=False
                        #
                        try:
                            current_wer = wer_score.compute(predictions=current_text_pred, references=current_text_true)
                            current_rouge = rouge_dot_safe(current_text_pred, current_text_true)
                            avg_wer += current_wer / len(val_loader)
                            avg_rouge += current_rouge / len(val_loader)
                        except:
                            print('Exception in metric')
                        #
                        # Save results for further processing
                        wers.append(current_wer)
                        rouges.append(current_rouge)
                        preds.append(predicted_ids.to('cpu').numpy().astype(np.int32))
                        text_pred.append(current_text_pred)
                        text_true.append(current_text_true)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    ROUGE: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_rouge, avg_wer, (time.time() - start)))
                        #
                # Compute mean metrics for VAL set in each epoch
                mean_wer = np.mean(wers)
                mean_rouge = np.mean(rouges)
                #
                # Save
                np.save(os.path.join(args.data_preds_dir, 'text_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id)), np.concatenate(text_pred))
                np.save(os.path.join(args.data_preds_dir, 'text_true_val_fold_%d_tta_%d.npy' % (fold_id, tta_id)), np.concatenate(text_true))
                with open(os.path.join(args.data_preds_dir, 'ids_pred_val_fold_%d_tta_%d.pkl' % (fold_id, tta_id)), 'wb') as f:   
                    pickle.dump(preds, f)
            #--------------------------------------------------------------------------
            # Predict TEST
            #--------------------------------------------------------------------------
            if 'test' in args.job:
                print('Predict TEST (fold %d, tta %d)' % (fold_id, tta_id))
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                avg_rouge = 0.0
                mean_loss = 0.0
                mean_wer = 0.0
                mean_rouge = 0.0
                losses = []
                wers = []
                rouges = []
                preds = []
                text_pred = []
                text_true = []
                #
                for i, (x_batch, zero_counts_batch) in enumerate(test_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    with torch.no_grad():
                        #
                        predicted_ids = model.generate(x_batch.to(device), max_length=args.max_len_out_infer, num_beams=args.n_beams_infer)
                        current_text_pred = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                        #
                        # Save results for further processing
                        preds.append(predicted_ids.to('cpu').numpy().astype(np.int32))
                        text_pred.append(current_text_pred)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    ROUGE: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_rouge, avg_wer, (time.time() - start)))
                        #
                # Save
                np.save(os.path.join(args.data_preds_dir, 'text_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id)), np.concatenate(text_pred))
                with open(os.path.join(args.data_preds_dir, 'ids_pred_test_fold_%d_tta_%d.pkl' % (fold_id, tta_id)), 'wb') as f:   
                    pickle.dump(preds, f)
                #
                test_df['Target'] = np.concatenate(text_pred)
                test_df[['ID', 'Target']].to_csv('text_pred_test_fold_%d_tta_%d_ewe.csv' % (fold_id, tta_id), index=False, encoding='utf-8')
        #--------------------------------------------------------------------------
        print('Cleaning...')
        try:
            del model
            del optimizer
            del loss
            del scaler
            del criterion
            del scheduler
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



