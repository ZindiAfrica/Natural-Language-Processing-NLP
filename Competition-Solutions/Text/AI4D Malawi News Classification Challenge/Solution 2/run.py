#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Import')

import os
import gc
import sys
sys.path.append('../../lib')
import glob
import math
from datetime import datetime as dt
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import tensorflow as tf
print('tf:', tf.__version__)
import transformers as tr
print('tr', tr.__version__)

from vecxoz_utils import init_tpu
from vecxoz_utils import init_tfdata_numpy
from vecxoz_utils import KeepLastCKPT
from vecxoz_utils import compute_cv_scores_cls_standalone
from vecxoz_utils import create_submission_cls_standalone
from vecxoz_utils import ArgumentParserExtended
from vecxoz_utils import seeder
from vecxoz_utils import create_cv_split

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Parse CMD')

parser = ArgumentParserExtended()

parser.add_str('--model_dir_or_name', default='google/mt5-large', help='Model name or directory containing weights and vocab')
parser.add_int('--max_len',           default=144,           help='Maximum sequence length')
parser.add_int('--seed',              default=748444,        help='Random seed')
parser.add_bool('--allow_growth',     default=True,          help='Whether to allow GPU mem growth (i.e. do not allocate all GPU mem right away)')
parser.add_int('--initial_fold',      default=None,          help='Initial fold (from 0)')
parser.add_int('--final_fold',        default=None,          help='Final fold (from 1)')
parser.add_str('--lib_dir',           default='../../lib',   help='Import directory')
parser.add_bool('--drop_remainder',   default=False,         help='Whether to drop remainder')
parser.add_int('--n_steps_train',     default=None,          help='Number of steps to train per epoch i.e. steps_per_epoch')
parser.add_int('--n_steps_train_val', default=None,          help='Number of steps to predict on val set during training')
parser.add_int('--n_steps_val',       default=None,          help='Number of steps to predict on val set for saving')
parser.add_int('--n_steps_test',      default=None,          help='Number of steps to predict on test set for saving')
parser.add_bool('--disable_eager',    default=False,         help='Whether to disable eager mode')
parser.add_bool('--skip_training',    default=False,         help='Whether to skip training for initial fold ONLY')

parser.add_str('--data_tfrec_dir',    default=None,          help='Data directory containig TFRecord files')
parser.add_str('--data_dir',          default='../../data',  help='Data directory containig CSV files')
parser.add_str('--data_preds_dir',    default='preds',       help='Directory inside working directory where predictions will be saved')
parser.add_str('--tpu_ip_or_name',    default=None,     help='TPU name or GRPC address e.g. node-1 or grpc://10.70.50.202:8470 or None')
parser.add_str('--mixed_precision',   default=None,          help='Mixed precision. E.g.: mixed_float16, mixed_bfloat16, or None')
parser.add_str('--job',               default='train_val_test_score_subm', help='Job to perform. A combination of words: train, val, test, score, subm. E.g. train_val_test')
parser.add_str('--metric_name',       default='acc',         help='Metric name')
parser.add_str('--monitor',           default='val_acc',     help='Value to monitor during training to apply checkpoint callback, etc.')
parser.add_int('--n_folds',           default=5,             help='Number of folds')
parser.add_int('--auto',              default=-1,            help='Constant value of tf.data.experimental.AUTOTUNE. It is used to manage number of parallel calls, etc.')
parser.add_int('--n_examples_train_total',  default=1436,          help='Number of training examples. This value is used to define an epoch')
parser.add_int('--n_epochs',          default=20,            help='Number of epochs to train')
parser.add_int('--batch_size',        default=24,           help='Batch size')
parser.add_float('--lr',              default=3e-5,          help='Learning rate')
parser.add_float('--aug_percentage',  default=0.5,           help='Probablity of outputting augmented image regardless of total number of augmentations')
parser.add_int('--aug_number',        default=0,             help='Number of train-time augmentations. 0 means no augmentation')
parser.add_int('--tta_number',        default=0,             help='Number of test-time augmentations. 0 means no augmentation. In result there will be (tta_number + 1) predictions')
parser.add_int('--n_classes',         default=20,             help='Number of classes for classification task. For regression task must be 1')
parser.add_float('--label_smoothing', default=0.1,           help='Label smoothing. Apllicable to classification task only')
parser.add_int('--buffer_size',       default=256,          help='Shuffle buffer size for tf.data. For small RAM or large data use small buffer e.g. 128 or None to disable')
parser.add_bool('--use_cache',        default=True,          help='Whether to use cache for tf.data')

args = parser.parse_args() # add args=[] to run in notebook

# Date-time
args.date_time=dt.now().strftime('%Y%m%d-%H%M%S-%f')

# Disable Eager mode
if args.disable_eager:
    tf.compat.v1.disable_eager_execution()

# Create dirs
if not os.path.exists(args.data_preds_dir):
    os.makedirs(args.data_preds_dir)

# Number of sub-train examples (e.g. 4/5 of full train)
args.n_examples_train = args.n_examples_train_total - (args.n_examples_train_total // args.n_folds)

# Train steps
if args.n_steps_train is None:
    args.n_steps_train = args.n_examples_train // args.batch_size

# Manage GPU MEM growth
if args.allow_growth:
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        except:
            print('Cannot set memory growth for device: %s. Already initialized?' % gpu.name)

# Set import dirs
if args.lib_dir:
    sys.path.append(args.lib_dir)

# Manage seed
if args.seed:
    print('Set seed:', seeder(seed=args.seed, general=True, te=True, to=False))
else:
    args.seed = np.random.randint(2**20)
    print('Set seed:', seeder(seed=args.seed, general=True, te=True, to=False))

# Folds
if args.initial_fold is None:
    args.initial_fold = 0
else:
    assert args.initial_fold <= args.n_folds - 1, 'Incorrect initial_fold'

if args.final_fold is None:
    args.final_fold = args.n_folds
else:
    assert args.final_fold <= args.n_folds and args.final_fold > args.initial_fold, 'Incorrect final_fold'

print('Settings')
print(parser.args_repr(args, False))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_model(print_summary=True, from_pretrained=True):
    """
    Init model with pretrained or random weights.
    Parameters:
        from_pretrained : bool, default True
            Set True for finetuning from petrained weights.
            Set False for training from scratch or inference with custom weights.
                This option allows to avoid automatic download of pretrained weights 
                before inference given that custom weights will be loaded anyway.
    """
    if from_pretrained:
        transformer = tr.TFMT5EncoderModel.from_pretrained(args.model_dir_or_name)
    else:
        config = tr.MT5Config.from_pretrained(args.model_dir_or_name)
        transformer = tr.TFMT5EncoderModel(config)
    input_ids = tf.keras.layers.Input(shape=(args.max_len,), dtype=tf.int32)
    sequence_output = transformer(input_ids)[0] # (batch, len, hidden)
    cls_token = sequence_output[:, 0, :] # (batch, hidden)
    out = tf.keras.layers.Dense(args.n_classes, activation='softmax')(cls_token)
    model = tf.keras.models.Model(inputs=input_ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=args.lr), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=[args.metric_name])
    if print_summary:
        model.summary()
    return model

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Load CSV and create CV split...')

train_df, test_df = create_cv_split(os.path.join(args.data_dir, 'Train.csv'), 
                                    os.path.join(args.data_dir, 'Test.csv'), 
                                    col_label='Label', 
                                    col_group=None, 
                                    n_folds=args.n_folds, 
                                    splitter='skf',
                                    random_state=33)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if args.job != 'score' and args.job != 'subm' and args.job != 'score_subm':

    print('Init tokenizer')
    
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_dir_or_name)
    print(tokenizer.__class__.__name__)
    
    print('Encode')
    
    encoded_dict_train = tokenizer.batch_encode_plus(
        list(train_df['Text'].values),
        max_length=args.max_len,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=False,)
    
    encoded_dict_test = tokenizer.batch_encode_plus(
        list(test_df['Text'].values),
        max_length=args.max_len,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=False,)
    
    X = np.array(encoded_dict_train['input_ids'], dtype=np.int32)
    y = train_df['Label_le'].values.astype(np.int32)
    X_test = np.array(encoded_dict_test['input_ids'], dtype=np.int32)
    fold_ids = train_df['fold_id'].values.astype(np.int32)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if args.job != 'score' and args.job != 'subm' and args.job != 'score_subm':
    for fold_id in range(args.initial_fold, args.final_fold):
        print('\n*****')
        print('Fold:', fold_id)
        print('*****\n')
        #--------------------------------------------------------------------------
        if args.mixed_precision is not None:
            print('Init Mixed Precision:', args.mixed_precision)
            policy = tf.keras.mixed_precision.experimental.Policy(args.mixed_precision)
            tf.keras.mixed_precision.experimental.set_policy(policy)
        else:
            print('Using default PRECISION:', tf.keras.backend.floatx())
        #--------------------------------------------------------------------------
        print('FULL BATCH SHAPE: %d x %d' % (args.batch_size,
                                             args.max_len,))
        #--------------------------------------------------------------------------
        print('Init TPU')
        tpu, topology, strategy = init_tpu(args.tpu_ip_or_name)
        #--------------------------------------------------------------------------
        print('Init datasets')
        X_train = X[fold_ids != fold_id]
        X_val = X[fold_ids == fold_id]
        #
        y_train = y[fold_ids != fold_id]
        y_val = y[fold_ids == fold_id]
        #
        train_ds = init_tfdata_numpy(X_train, y_train,
                                     is_train=True, 
                                     batch_size=args.batch_size, 
                                     auto=args.auto, 
                                     buffer_size=args.buffer_size,
                                     use_cache=args.use_cache)
        val_ds = init_tfdata_numpy(X_val, y_val,
                                   is_train=False,  
                                   batch_size=args.batch_size, 
                                   auto=args.auto,
                                   buffer_size=args.buffer_size,
                                   use_cache=args.use_cache)
        #--------------------------------------------------------------------------
        print('Init model')
        with strategy.scope():
            model = init_model(print_summary=True, from_pretrained='train' in args.job)
        #--------------------------------------------------------------------------
        print('Init callbacks')
        call_ckpt = tf.keras.callbacks.ModelCheckpoint('model-best-f%d-e{epoch:03d}-{val_%s:.4f}.h5' % (fold_id, args.metric_name),
                                                       monitor=args.monitor, # do not use if no val set
                                                       save_best_only=True, # set False if no val set
                                                       save_weights_only=True,
                                                       mode='auto',
                                                       verbose=1)
        call_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor, 
                                                              factor=0.5, 
                                                              patience=3, 
                                                              min_delta=1e-4,
                                                              min_lr=1e-8,
                                                              verbose=1,
                                                              mode='auto')
        call_keep_last = KeepLastCKPT(wildcard = './model-best-f%d-e*.h5' % fold_id)
        #-------------------------------------------------------------------------- 
        if 'train' in args.job and not args.skip_training:
            args.skip_training = False # do not skip for all next folds
            print('Fit (fold %d)' % fold_id)
            h = model.fit(
                train_ds,
                # batch_size=args.batch_size, # just not to forget
                steps_per_epoch=args.n_steps_train,
                validation_steps=args.n_steps_train_val,
                epochs=args.n_epochs,
                validation_data=val_ds,
                callbacks=[call_ckpt,
                           call_reduce_lr,
                           call_keep_last]
            )
        #--------------------------------------------------------------------------
        # Load best model for fold
        m = sorted(glob.glob('model-best-f%d*.h5' % fold_id))[-1]
        print('Load model (fold %d): %s' % (fold_id, m))
        model.load_weights(m)
        #--------------------------------------------------------------------------
        # TTA
        #--------------------------------------------------------------------------
        for tta_id in range(args.tta_number + 1):
            # Create VAL and TEST datasets with TTA transforms corresponding to AUG transforms seen during training
            print('Init datasets for prediction (fold %d, tta %d)' % (fold_id, tta_id))
            val_ds = init_tfdata_numpy(X_val, 
                                       is_train=False,  
                                       batch_size=args.batch_size, 
                                       auto=args.auto,
                                       buffer_size=args.buffer_size,
                                       use_cache=args.use_cache)
            test_ds = init_tfdata_numpy(X_test, 
                                        is_train=False,  
                                        batch_size=args.batch_size, 
                                        auto=args.auto,
                                        buffer_size=args.buffer_size,
                                        use_cache=args.use_cache)
            #--------------------------------------------------------------------------
            # Predict val
            if 'val' in args.job:
                print('Predict VAL (fold %d, tta %d)' % (fold_id, tta_id))
                y_pred_val = model.predict(val_ds, verbose=1, steps=args.n_steps_val)
                np.save(os.path.join(args.data_preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id)), y_pred_val)
            #--------------------------------------------------------------------------
            # Predict test
            if 'test' in args.job:
                print('Predict TEST (fold %d, tta %d)' % (fold_id, tta_id))
                y_pred_test = model.predict(test_ds, verbose=1, steps=args.n_steps_test)
                np.save(os.path.join(args.data_preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id)), y_pred_test)
            #--------------------------------------------------------------------------
        print('Cleaning...')
        try:
            del tpu
            del topology
            del strategy
            del train_ds
            del val_ds
            del test_ds
            del model
            del call_ckpt
            del call_reduce_lr
            del call_keep_last
            del h
        except:
            pass
        gc.collect()

#------------------------------------------------------------------------------
# Compute val scores
#------------------------------------------------------------------------------

if 'val' in args.job or 'score' in args.job:
    print('VAL scores')
    scores = compute_cv_scores_cls_standalone(args.data_dir, args.data_preds_dir, args.n_folds, args.tta_number, print_scores=True)

#------------------------------------------------------------------------------
# Create submission
#------------------------------------------------------------------------------

if 'test' in args.job or 'subm' in args.job:
    print('Create submission CSV')
    written_file_name = create_submission_cls_standalone(args.data_dir, args.data_preds_dir, args.n_folds, args.tta_number)
    print('Submission was saved to:', written_file_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------





