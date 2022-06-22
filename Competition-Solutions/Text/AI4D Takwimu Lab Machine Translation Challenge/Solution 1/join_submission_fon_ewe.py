#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import pandas as pd

print('Joining Fon and Ewe parts...')

fon_df = pd.read_csv('models/run-20210531-0044-to-mt5b-finetune-fon/text_pred_test_fold_0_tta_0_fon.csv')
eve_df = pd.read_csv('models/run-20210531-0046-to-mt5b-finetune-ewe/text_pred_test_fold_0_tta_0_ewe.csv')
final_df = pd.concat([fon_df, eve_df])

final_df.to_csv('submission-final.csv', index=False, encoding='utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
