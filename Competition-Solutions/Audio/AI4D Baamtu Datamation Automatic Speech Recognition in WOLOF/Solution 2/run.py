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
import torchaudio
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Parse CMD...')

parser = ArgumentParserExtended()

parser.add_str('--model_dir_or_name', default='facebook/wav2vec2-large-xlsr-53', help='Model name or directory containing weights and vocab')
parser.add_int('--max_len',           default=147840,        help='Maximum sequence length for audio')
parser.add_int('--max_len_label',     default=60,            help='Maximum sequence length for text (transcription)')
parser.add_int('--seed',              default=286180,        help='Random seed global')
parser.add_int('--seed_for_data',     default=33,            help='Random seed for data split')
parser.add_int('--accumulation_steps',default=2,             help='Accumulation steps')
parser.add_str('--device',            default='cuda:0',      help='Data directory containig TFRecord files')
parser.add_int('--print_n_batches',   default=33,            help='Report iterval')
parser.add_str('--out_dir',           default='./',          help='Data directory containig CSV files')
parser.add_bool('--torch_mixed_precision', default=True,     help='Whether to use torch.cuda.amp')
parser.add_int('--initial_fold',      default=0,             help='Initial fold')
parser.add_int('--final_fold',        default=1,             help='Final fold (from 1). None means all folds i.e. equal to n_folds')
parser.add_str('--lib_dir',           default='../../lib',      help='Import directory')
parser.add_int('--n_steps_train',     default=None,          help='Number of steps to train per epoch i.e. steps_per_epoch. None means all examples')
parser.add_str('--torch_monitor',     default='val_wer',     help='Value to monitor during training to apply checkpoint callback, etc.')
parser.add_str('--torch_sheduler_mode', default='min',       help='Value to monitor during training to apply checkpoint callback, etc.')
parser.add_bool('--skip_training',    default=False,         help='Whether to skip training for initial fold ONLY')
parser.add_bool('--freeze',           default=True,          help='Whether to freeze feature extractor')
parser.add_bool('--use_lr_schedule',  default=True,          help='Whether to use LR schedule')

parser.add_str('--data_dir',          default='../../data', help='Data directory containig CSV files')
parser.add_str('--data_preds_dir',    default='preds',       help='Directory inside working directory where predictions will be saved')
parser.add_str('--job',               default='train_test', help='Job to perform. A combination of words: train, val, test, score, subm. E.g. train_val_test')
parser.add_int('--n_folds',           default=100,           help='Number of folds')
parser.add_int('--auto',              default=-1,            help='Constant value of tf.data.experimental.AUTOTUNE. It is used to manage number of parallel calls, etc.')
parser.add_int('--n_examples_train_total',  default=6683,    help='Number of training examples. This value is used to define an epoch')
parser.add_int('--n_epochs',          default=296,           help='Number of epochs to train')
parser.add_int('--batch_size',        default=16,            help='Batch size')
parser.add_int('--batch_size_infer',  default=1,             help='Batch size for inference')
parser.add_float('--lr',              default=1e-4,          help='Learning rate')
parser.add_float('--aug_percentage',  default=0.5,           help='Probablity of outputting augmented image regardless of total number of augmentations')
parser.add_int('--aug_number',        default=0,             help='Number of train-time augmentations. 0 means no augmentation')
parser.add_int('--tta_number',        default=0,             help='Number of test-time augmentations. 0 means no augmentation. In result there will be (tta_number + 1) predictions')
parser.add_int('--buffer_size',       default=256,           help='Shuffle buffer size for tf.data. For small RAM or large data use small buffer e.g. 128 or None to disable')
parser.add_bool('--use_cache',        default=True,          help='Whether to use cache for tf.data')

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

print('Definitions...')

def ecode_audio_numpy_padded(df, sampling_rate=16000, max_length=args.max_len, verboze=True):
    n_examples = df.shape[0]
    encoded = []
    for counter, file_id in enumerate(df['ID']):
        audio, rate = torchaudio.load(os.path.join(args.data_dir, 'clips', file_id + '.mp3'))
        audio = resampler(audio).squeeze().numpy()
        # audio, rate = librosa.load(os.path.join(args.data_dir, 'clips', file_id + '.mp3'), sr=16000)
        res = processor(audio, 
                        sampling_rate=sampling_rate, 
                        return_tensors='np', 
                        padding='max_length', 
                        max_length=max_length
                        )
        input_values = res.input_values
        encoded.append(input_values[0])
        if not counter % 500 and verboze:
            print(counter, 'of', n_examples)
    X = np.array(encoded, dtype=np.float32)
    return X


def ecode_text_numpy_padded(df, max_length=args.max_len_label, verboze=True):
    n_examples = df.shape[0]
    encoded = []
    for counter, text in enumerate(df['transcription']):
        with processor.as_target_processor():
            res = processor(text, 
                            return_tensors='np', 
                            padding='max_length', 
                            max_length=max_length,
                            )
        input_ids = res.input_ids
        encoded.append(input_ids[0])
        if not counter % 500 and verboze:
            print(counter, 'of', n_examples)
    X = np.array(encoded, dtype=np.int32)
    return X


def filter_special(x):
    x = x.lower()
    return x

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Load CSV and create CV split...')

train_df, test_df = create_cv_split(os.path.join(args.data_dir, 'Train.csv'), 
                                    os.path.join(args.data_dir, 'Test.csv'), 
                                    col_label='transcription', 
                                    col_group=None, 
                                    n_folds=args.n_folds, 
                                    splitter='kf',
                                    random_state=args.seed_for_data)

print(train_df['transcription'].iloc[0]) # Autoroute seydina-limamoulaye

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Create vocab and init tokenizer')

# Filter special
train_df['transcription'] = train_df['transcription'].map(filter_special)
# Create vocab
global_string = ' '.join(train_df['transcription'].tolist())
vocab_list = sorted(list(set(global_string)))
# Add special tokens
vocab_list = ["[PAD]", "[UNK]", "|"] + vocab_list
# Remove space (because we will use pipe "|" insated)
vocab_list.remove(" ")
# Create letter ids
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
# print(len(vocab_dict)) # 44
# Save
with open('vocab.json', 'w') as f:
    json.dump(vocab_dict, f)

print('Init tokenizer')

tokenizer = tr.Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = tr.Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = tr.Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
resampler = torchaudio.transforms.Resample(48000, 16000)

print('Processor:        ', processor.__class__.__name__)                   # Wav2Vec2Processor
print('Feature extractor:', processor.feature_extractor.__class__.__name__) # Wav2Vec2FeatureExtractor
print('Tokenizer:        ', processor.tokenizer.__class__.__name__)         # Wav2Vec2CTCTokenizer

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if 'train' in args.job or 'val' in args.job:
    print('Encoding training data...')
    X = ecode_audio_numpy_padded(train_df)
    Y = ecode_text_numpy_padded(train_df)    
    # Replace padding token ID with -100 according to Huggingface implementation
    # https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
    Y[Y==processor.tokenizer.pad_token_id] = -100 # <-------------------------------------------- IMPORTANT
    fold_ids = train_df['fold_id'].values.astype(np.int32)

if 'test' in args.job:
    print('Encoding test data...')
    X_test = ecode_audio_numpy_padded(test_df)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Enter fold loop...')

if args.job != 'score' and args.job != 'subm' and args.job != 'score_subm':
    for fold_id in range(args.initial_fold, args.final_fold):
        print('\n*****')
        print('Fold:', fold_id)
        print('*****\n')
        #--------------------------------------------------------------------------
        best_score = 100
        start = None
        #--------------------------------------------------------------------------
        if args.torch_mixed_precision:
            print('Using AMP')
        else:
            print('Using default precision')
        # Create fresh instance for each fold
        scaler = torch.cuda.amp.GradScaler(enabled=args.torch_mixed_precision)
        #--------------------------------------------------------------------------
        print('FULL TRAIN BATCH SHAPE: %d x %d' % (args.batch_size,
                                                    args.max_len,))
        print('FULL INFER BATCH SHAPE: %d x %d' % (args.batch_size_infer,
                                                    args.max_len,))        
        #--------------------------------------------------------------------------
        print('Init model')
        # For finetuning we init from pretrained to load Huggingface pretrained weights
        # For inference (when we will load our own finetuned model) we init from config 
        #   to avoid downloading Huggingface pretrained weights
        if 'train' in args.job:
            model = tr.Wav2Vec2ForCTC.from_pretrained(
                      args.model_dir_or_name, # "facebook/wav2vec2-large-xlsr-53",
                      attention_dropout=0.1,
                      hidden_dropout=0.1,
                      feat_proj_dropout=0.0,
                      mask_time_prob=0.05,
                      layerdrop=0.1,
                      gradient_checkpointing=True,
                      ctc_loss_reduction='mean',
                      pad_token_id=processor.tokenizer.pad_token_id,
                      vocab_size=len(processor.tokenizer))
        else:
            config = tr.Wav2Vec2Config.from_pretrained(
                      args.model_dir_or_name, # "facebook/wav2vec2-large-xlsr-53",
                      attention_dropout=0.1,
                      hidden_dropout=0.1,
                      feat_proj_dropout=0.0,
                      mask_time_prob=0.05,
                      layerdrop=0.1,
                      gradient_checkpointing=True,
                      ctc_loss_reduction='mean',
                      pad_token_id=processor.tokenizer.pad_token_id,
                      vocab_size=len(processor.tokenizer))
            model = tr.Wav2Vec2ForCTC(config)
        #--------------------------------------------------------------------------
        if args.freeze:
            model.freeze_feature_extractor() # FREEZE
        #--------------------------------------------------------------------------
        device = torch.device(args.device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.use_lr_schedule:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=args.torch_sheduler_mode, factor=0.5, patience=7, verbose=True)
        #-------------------------------------------------------------------------- 
        # TRAIN
        #--------------------------------------------------------------------------
        if 'train' in args.job and not args.skip_training:
            print('Enter training loop...')
            print('Fit (fold %d)' % fold_id)
            #
            print('Init NPY datasets...')
            X_train = X[fold_ids != fold_id]
            X_val = X[fold_ids == fold_id]
            Y_train = Y[fold_ids != fold_id]
            Y_val = Y[fold_ids == fold_id]
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
                X_train, Y_train, zero_counts_train, ids_back_train = prepare_trim(X_train[shuffle_ids], Y_train[shuffle_ids], pad_token_id=processor.feature_extractor.padding_value)
                # Stage-2: shuffle batches
                #          this shuffle will preserve same (close) zero count in each batch
                shuffle_ids = shuffle_batches(X_train, batch_size=args.batch_size)
                td_train = torch.utils.data.TensorDataset(torch.tensor(X_train[shuffle_ids], dtype=torch.float32),
                                                          torch.tensor(Y_train[shuffle_ids], dtype=torch.long),
                                                          torch.tensor(zero_counts_train[shuffle_ids], dtype=torch.long))
                train_loader = torch.utils.data.DataLoader(td_train, batch_size=args.batch_size, shuffle=False)
                #
                # VAL
                X_val, Y_val, zero_counts_val, ids_back_val = prepare_trim(X_val, Y_val, pad_token_id=processor.feature_extractor.padding_value)
                td_val = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                torch.tensor(Y_val, dtype=torch.long),
                                                torch.tensor(zero_counts_val, dtype=torch.long))
                val_loader = torch.utils.data.DataLoader(td_val, batch_size=args.batch_size, shuffle=False)
                #--------------------------------------------------------------------------
                #
                avg_loss = 0
                avg_wer = 0
                #
                model.train()
                optimizer.zero_grad()
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(train_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    #
                    with torch.cuda.amp.autocast(enabled=args.torch_mixed_precision):
                        res = model(x_batch.to(device), 
                                    labels=y_batch.to(device), 
                                    attention_mask=(x_batch != processor.feature_extractor.padding_value).long().to(device))
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
                    # According to Huggingface implementation https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
                    predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
                    y_batch[y_batch == -100] = processor.tokenizer.pad_token_id
                    #
                    current_text_pred = processor.batch_decode(predicted_ids)
                    current_text_true = processor.batch_decode(y_batch, group_tokens=False)
                    #
                    avg_wer += wer_score.compute(predictions=current_text_pred, references=current_text_true) / len(train_loader)
                    #
                    if not i % args.print_n_batches:
                        print('Batch: %6d    Loss: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_wer, (time.time() - start)))
                #
                #--------------------------------------------------------------------------
                # EVAL after each epoch
                #--------------------------------------------------------------------------
                print('Eval...')
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                losses = []
                wers = []
                preds = []
                text_pred = []
                text_true = []
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(val_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=args.torch_mixed_precision):
                            res = model(x_batch.to(device), 
                                        labels=y_batch.to(device), 
                                        attention_mask=(x_batch != processor.feature_extractor.padding_value).long().to(device))
                        loss = res.loss
                        logits = res.logits
                        #
                        # Val metrics
                        avg_loss += loss.item() / len(val_loader)
                        predicted_ids = torch.argmax(logits, dim=-1)
                        #
                        # According to Huggingface implementation https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
                        predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
                        y_batch[y_batch == -100] = processor.tokenizer.pad_token_id
                        #
                        current_text_pred = processor.batch_decode(predicted_ids)
                        current_text_true = processor.batch_decode(y_batch, group_tokens=False)
                        #
                        currect_wer = wer_score.compute(predictions=current_text_pred, references=current_text_true)
                        avg_wer += currect_wer / len(val_loader)
                        #
                        # Save results for further processing
                        losses.append(loss.item())
                        wers.append(currect_wer)
                        preds.append(logits.to('cpu').numpy().astype(np.float32))
                        text_pred.append(current_text_pred)
                        text_true.append(current_text_true)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_wer, (time.time() - start)))
                        #
                # Compute mean metrics for VAL set in each epoch
                mean_loss = np.mean(losses)
                mean_wer = np.mean(wers)
                # Save model if score is better
                print('MEAN val LOSS: %.6f    MEAN val WER: %.6f' % (mean_loss, mean_wer))
                print('Epoch time: %6d sec' % (time.time() - start))
                if mean_wer <= best_score:
                    print('Saving model (%.6f <= %.6f)...' % (mean_wer, best_score))
                    best_score = mean_wer
                    p = os.path.join(args.out_dir, 'model-best-f%d-e%03d-%.4f.bin' % (fold_id, epoch, mean_wer))
                    torch.save(model.state_dict(), p)
                    print('Saved model:', p)
                else:
                    print('Mean WER is not better (%.6f > %.6f): NOT saving the model' % (mean_wer, best_score))
                #
                # Remove all previous checkpoints
                keep_last_ckpt_torch(os.path.join(args.out_dir, 'model-best-f%d-e*.bin' % fold_id))
                #
                if args.use_lr_schedule:
                    if args.torch_monitor == 'val_wer':
                        scheduler.step(mean_wer)
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
            
            # For inference we use regular trimming without grouping by sequence length

            if 'val' in args.job:
                X_val = X[fold_ids == fold_id]
                Y_val = Y[fold_ids == fold_id]

                zero_counts_val = np.sum(X_val == processor.feature_extractor.padding_value, axis=1)
                td_val = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                                        torch.tensor(Y_val, dtype=torch.long), 
                                                        torch.tensor(zero_counts_val, dtype=torch.long),)
                val_loader = torch.utils.data.DataLoader(td_val, batch_size=args.batch_size_infer, shuffle=False)

            if 'test' in args.job:
                zero_counts_test = np.sum(X_test == processor.feature_extractor.padding_value, axis=1)
                td_test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),  
                                                         torch.tensor(zero_counts_test, dtype=torch.long),)
                test_loader = torch.utils.data.DataLoader(td_test, batch_size=args.batch_size_infer, shuffle=False)

            #--------------------------------------------------------------------------
            # Predict VAL
            #--------------------------------------------------------------------------
            if 'val' in args.job:
                print('Predict VAL (fold %d, tta %d)' % (fold_id, tta_id))
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                losses = []
                wers = []
                preds = []
                text_pred = []
                text_true = []
                #
                for i, (x_batch, y_batch, zero_counts_batch) in enumerate(val_loader):
                    # Find min zero count for given batch
                    min_zero_count = zero_counts_batch.min()
                    # Trim batch
                    x_batch = x_batch[:, :(args.max_len - min_zero_count)]
                    with torch.no_grad():
                        res = model(x_batch.to(device), 
                                    labels=y_batch.to(device), 
                                    attention_mask=(x_batch != processor.feature_extractor.padding_value).long().to(device))
                        loss = res.loss
                        logits = res.logits
                        #
                        # Val metrics
                        avg_loss += loss.item() / len(val_loader)
                        predicted_ids = torch.argmax(logits, dim=-1)
                        #
                        # According to Huggingface implementation https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
                        predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
                        y_batch[y_batch == -100] = processor.tokenizer.pad_token_id
                        #
                        current_text_pred = processor.batch_decode(predicted_ids)
                        current_text_true = processor.batch_decode(y_batch, group_tokens=False)
                        #
                        currect_wer = wer_score.compute(predictions=current_text_pred, references=current_text_true)
                        avg_wer += currect_wer / len(val_loader)
                        #
                        # Save results for further processing
                        losses.append(loss.item())
                        wers.append(currect_wer)
                        preds.append(logits.to('cpu').numpy().astype(np.float32))
                        text_pred.append(current_text_pred)
                        text_true.append(current_text_true)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_wer, (time.time() - start)))
                        #
                # Compute mean metrics for VAL set for the best epoch
                mean_loss = np.mean(losses)
                mean_wer = np.mean(wers)
                np.save(os.path.join(args.data_preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id)), np.array(text_pred))
            #--------------------------------------------------------------------------
            # Predict TEST
            #--------------------------------------------------------------------------
            if 'test' in args.job:
                print('Predict TEST (fold %d, tta %d)' % (fold_id, tta_id))
                model.eval()
                avg_loss = 0.0
                avg_wer = 0.0
                losses = []
                wers = []
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
                        res = model(x_batch.to(device), 
                                    labels=None, 
                                    attention_mask=(x_batch != processor.feature_extractor.padding_value).long().to(device))
                        logits = res.logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        #
                        # According to Huggingface implementation https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
                        predicted_ids[predicted_ids == -100] = processor.tokenizer.pad_token_id
                        #
                        current_text_pred = processor.batch_decode(predicted_ids)
                        #
                        # Save results for further processing
                        preds.append(logits.to('cpu').numpy().astype(np.float32))
                        text_pred.append(current_text_pred)
                        #
                        if not i % args.print_n_batches:
                            print('Batch: %6d    Loss: %.6f    WER: %.6f    Time: %6d sec' % (i, avg_loss, avg_wer, (time.time() - start)))
                        #
                # Compute mean metrics for VAL set in each epoch
                text_pred = np.array(text_pred).ravel()
                np.save(os.path.join(args.data_preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id)), text_pred)
                # Create submission for the current fold
                test_df['transcription'] = text_pred
                test_df.loc[test_df['transcription'] == '', 'transcription'] = 'a'
                test_df = test_df.fillna('a')
                test_df[['ID', 'transcription']].to_csv('../../submission-raw.csv', index=False)
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


