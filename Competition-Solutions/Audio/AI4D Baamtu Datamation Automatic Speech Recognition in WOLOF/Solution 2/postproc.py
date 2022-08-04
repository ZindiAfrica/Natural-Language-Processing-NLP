#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from editdistance import eval as levenstein_score

print('Post-processing...')

data_dir = './data'

train_df = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
subm_df = pd.read_csv(os.path.join('./', 'submission-raw.csv'))

train_df['transcription'] = train_df['transcription'].map(str.lower)

lines_train = train_df['transcription'].values
lines_pred = subm_df['transcription'].values

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def pp_lev(line_p, verboze=False):
    line_to_return = line_p
    #
    line_best_lev = line_p
    score_best_lev = 100
    #
    if line_p not in lines_train:        
        for line_t in lines_train:
            #
            score = levenstein_score(line_t, line_p)
            if score < score_best_lev:
                score_best_lev = score
                line_best_lev = line_t
            #
        if score_best_lev < 25:
            line_to_return = line_best_lev
    #
    return line_to_return

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

subm_df['transcription'] = subm_df['transcription'].map(pp_lev)

subm_df[['ID', 'transcription']].to_csv('submission-final.csv', index=False, encoding='utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



