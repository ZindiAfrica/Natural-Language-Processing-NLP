#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import re
import glob
import math
import json
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers as tr
import torch
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ArgumentParserExtended(ArgumentParser):
    """
    The main purpose of this class is to standardize and simplify definition of arguments
    and allow processing of True, False, and None values.
    There are 4 types of arguments (bool, int, float, str). All accept None.
    
    Usage:

    parser = ArgumentParserExtended()
    
    parser.add_str('--str', default='/home/user/data')
    parser.add_int('--int', default=220)
    parser.add_float('--float', default=3.58)
    parser.add_bool('--bool', default=True)
    
    args = parser.parse_args()
    print(parser.args_repr(args, True))
    """

    def __init__(self, *args, **kwargs):
        super(ArgumentParserExtended, self).__init__(*args, **kwargs)

    def bool_none_type(self, x):
        if x == 'True':
            return True
        elif x == 'False':
            return False
        elif x == 'None':
            return None
        else:
            raise ValueError('Unexpected literal for bool type')

    def int_none_type(self, x):
        return None if x == 'None' else int(x)

    def float_none_type(self, x):
        return None if x == 'None' else float(x)

    def str_none_type(self, x):
        return None if x == 'None' else str(x)

    def add_str(self, name, default=None, choices=None, help='str or None'):
        """
        Returns str or None
        """
        _ = self.add_argument(name, type=self.str_none_type, default=default, choices=choices, help=help)

    def add_int(self, name, default=None, choices=None, help='int or None'):
        """
        Returns int or None
        'hello' or 'none' or 1.2 will cause an error
        """
        _ = self.add_argument(name, type=self.int_none_type, default=default, choices=choices, help=help)

    def add_float(self, name, default=None, choices=None, help='float or None'):
        """
        Returns float or None
        'hello' or 'none' will cause an error
        """
        _ = self.add_argument(name, type=self.float_none_type, default=default, choices=choices, help=help)

    def add_bool(self, name, default=None, help='bool'):
        """
        Returns True, False, or None
        Anything except 'True' or 'False' or 'None' will cause an error

        `choices` are checked after type conversion of argument passed in fact
            i.e. `choices` value must be True instead of 'True'
        Default value is NOT checked using `choices`
        Default value is NOT converted using `type`
        """
        _ = self.add_argument(name, type=self.bool_none_type, default=default, choices=[True, False, None], help=help)

    @staticmethod
    def args_repr(args, print_types=False):
        ret = ''
        props = vars(args)
        keys = sorted([key for key in props])
        vals = [str(props[key]) for key in props]
        max_len_key = len(max(keys, key=len))
        max_len_val = len(max(vals, key=len))
        if print_types:
            for key in keys:
                ret += '%-*s  %-*s  %s\n' % (max_len_key, key, max_len_val, props[key], type(props[key]))
        else:   
            for key in keys:
                ret += '%-*s  %s\n' % (max_len_key, key, props[key])
        return ret.rstrip()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def seeder(seed, general=True, te=True, to=True):
    """
    seed : int or None
        None means do not set seed
    """
    if seed is not None:
        if general:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
        if te:
            tf.random.set_seed(seed)
        if to:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def prepare_trim(X, y=None, pad_token_id=0):
    """
    Sort data by the number of padding tokens (zeros).
    Each batch of sorted data will contain equal or close number of padding tokens.
    This makes it possible to efficiently trim padding tokens for each batch like the following:

    min_zero_count = zero_counts_batch.min()
    x_batch = x_batch[:, :(max_len - min_zero_count)]

    `ids_back` allows to get initial order of data if needed.
    """
    zero_counts = np.sum(X == pad_token_id, axis=1)
    ids_sorted = np.argsort(zero_counts)
    zero_counts_sorted = zero_counts[ids_sorted]
    X_sorted = X[ids_sorted]
    if y is not None:
        y_sorted = y[ids_sorted]
    else:
        y_sorted = None
    ids_back = np.argsort(ids_sorted)
    return X_sorted, y_sorted, zero_counts_sorted, ids_back

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def shuffle_batches(X, batch_size):
    """
    Creates indices to shuffle blocks of data (not individual examples).
    Such shuffle is useful when data is sorted for some purpose e.g. for trimming.
    In result order of examples are preserved within each batch.

    Usage:
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],
                  [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7],
                  [8, 8, 8], [9, 9, 9]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    shuffle_ids = shuffle_batches(X, batch_size=4)
    X[shuffle_ids]
    y[shuffle_ids]
    """
    n_batches = np.ceil(X.shape[0] / batch_size).astype(np.int32)
    indices = np.arange(X.shape[0])
    indices_per_batch = []
    for i in range(n_batches):
        indices_per_batch.append(indices[i*batch_size : (i+1)*batch_size].copy())
    np.random.shuffle(indices_per_batch)
    return np.concatenate(indices_per_batch)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class CustomSequenceClassifierOutput():
    def __init__(self, logits):
        self.logits = logits

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_cv_split(file_train, file_test, col_label='label', col_group=None, n_folds=5, splitter='skf', random_state=33):
    """
    Parameters:
        splitter : str
            "kf", "skf", "gkf"
    Example:
        train_df, test_df = create_cv_split(os.path.join(args.data_dir, 'Train.csv'), 
                                            os.path.join(args.data_dir, 'Test.csv'), 
                                            col_label='Label', 
                                            col_group=None, 
                                            n_folds=5, 
                                            splitter='skf',
                                            random_state=33)
    """
    #
    # In KFold and StratifiedKFold "groups" are always ignored
    # so we just make substitute to unify split call
    if col_group is None:
        col_group = col_label

    train_df = pd.read_csv(file_train)
    test_df = pd.read_csv(file_test)
    #
    # Label encoded label
    le = LabelEncoder()
    train_df[col_label + '_le'] = le.fit_transform(train_df[col_label])
    
    # Fake label for test (just for compatibility)
    test_df[col_label] = 0
    test_df[col_label + '_le'] = 0
    # Template column for fold_id
    train_df['fold_id'] = 0
    test_df['fold_id'] = 0 # (just for compatibility)
    # Check train/test columns
    assert list(train_df.columns) == list(test_df.columns), 'Different set or order of columns in train/test'
    
    if splitter == 'kf':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif splitter == 'skf':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif splitter == 'gkf':
        kf = GroupKFold(n_splits=n_folds)
    else:
        raise ValueError('Posible values for splitter are: "kf", "skf", and "gkf"')

    for fold_id, (train_index, val_index) in enumerate(kf.split(X=train_df, y=train_df[col_label].values, groups=train_df[col_group].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id

    # Check fold_id: must have corresponding number of folds
    assert len(train_df['fold_id'].unique()) == n_folds, 'Inconsistent number of folds'
    # Check fold_id: must be consequtive and start from 0
    lst = list(train_df['fold_id'])
    assert list(np.sort(np.unique(lst))) == list(range(0, max(lst)+1)), 'Non-consequtive, or starts not from 0'
    # Check groups: must not intersect
    if splitter == 'gkf':
        for i in range(n_folds):
            assert train_df[train_df['fold_id'] == i][col_group].isin(train_df[train_df['fold_id'] != i][col_group]).sum() == 0, 'Groups are intersected'

    # Shuffle
    # We use random_state+1 because 'df.sample' with the same seed after 'KFold.split' will re-create initial order
    train_df = train_df.sample(frac=1.0, random_state=random_state+1)
    #
    return train_df, test_df

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def keep_last_ckpt_torch(wildcard):
    """
    Sort all ckpt files matching the wildcard and remove all except last.
    If there is only one ckpt file it will not be removed.
    If naming is consistent e.g. "model-best-f0-e001-25.3676.h5"
        then KeepLastCKPT will keep OVERALL best ckpt

    Example:
    Call this function on epoch end.
    File template for weights:
        weight_name = 'model-best-f%d-e%03d-%.4f.bin' % (fold_id, epoch, score)
    Call:
        keep_last_ckpt_torch('./model-best-f%d-e*.bin' % fold_id)
    """
    files = sorted(glob.glob(wildcard))
    # files = sorted(tf.io.gfile.glob(wildcard))
    if len(files):
        for file in files[:-1]:
            os.remove(file)
            # tf.io.gfile.remove(file)
        print('Kept ckpt: %s' % files[-1])
    else:
        print('No ckpt to keep')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------













