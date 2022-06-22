#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import pandas as pd

print('Split original competition data into Fon and Ewe subsets...')

train_df = pd.read_csv('data/Train.csv')
test_df = pd.read_csv('data/Test.csv')

train_fon_df = train_df[train_df['Target_Language'] == 'Fon'].copy() 
print('Fon train:', train_fon_df.shape) # (53134, 4)
train_fon_df.to_csv('data/Train_Fon.csv', index=False, encoding='utf-8')

train_ewe_df = train_df[train_df['Target_Language'] == 'Ewe'].copy() 
print('Ewe train:', train_ewe_df.shape) # (22353, 4)
train_ewe_df.to_csv('data/Train_Ewe.csv', index=False, encoding='utf-8')

test_fon_df = test_df[test_df['Target_Language'] == 'Fon'].copy() 
print('Fon test:', test_fon_df.shape) # (2929, 3)
test_fon_df.to_csv('data/Test_Fon.csv', index=False, encoding='utf-8')

test_ewe_df = test_df[test_df['Target_Language'] == 'Ewe'].copy()
print('Ewe test:', test_ewe_df.shape) # (2964, 3)
test_ewe_df.to_csv('data/Test_Ewe.csv', index=False, encoding='utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

