#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import pandas as pd

fra_fon__fra_lines = []
with open('fon/train-parts/JW300-fra_fon.fra', 'r') as f:
    for line in f:
        fra_fon__fra_lines.append(line.strip())

fra_fon__fon_lines = []
with open('fon/train-parts/JW300-fra_fon.fon', 'r') as f:
    for line in f:
        fra_fon__fon_lines.append(line.strip())

fra_ewe__fra_lines = []
with open('ewe/train-parts/JW300-fra_ewe.fra', 'r') as f:
    for line in f:
        fra_ewe__fra_lines.append(line.strip())

fra_ewe__ewe_lines = []
with open('ewe/train-parts/JW300-fra_ewe.ewe', 'r') as f:
    for line in f:
        fra_ewe__ewe_lines.append(line.strip())


fra_fon_df = pd.DataFrame()
fra_fon_df['French'] = fra_fon__fra_lines
fra_fon_df['Target'] = fra_fon__fon_lines
print(fra_fon_df.shape) # (31962, 2)
fra_fon_df.to_csv('jw300_fra_fon.csv', index=False, encoding='utf-8')

fra_ewe_df = pd.DataFrame()
fra_ewe_df['French'] = fra_ewe__fra_lines
fra_ewe_df['Target'] = fra_ewe__ewe_lines
print(fra_ewe_df.shape) # (611204, 2)
fra_ewe_df.to_csv('jw300_fra_ewe.csv', index=False, encoding='utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

fra_fon_df['Target_Language'] = 'Fon'
fra_ewe_df['Target_Language'] = 'Ewe'

fra_fon_df['ID'] = 'no_id'
fra_ewe_df['ID'] = 'no_id'

train_fon_df = pd.read_csv('Train_Fon.csv')
train_ewe_df = pd.read_csv('Train_Ewe.csv')

all_fon_df = pd.concat([train_fon_df, fra_fon_df])
print(all_fon_df.shape) # (85096, 4)
all_ewe_df = pd.concat([train_ewe_df, fra_ewe_df])
print(all_ewe_df.shape) # (633557, 4)

all_fon_df = all_fon_df.sample(frac=1.0, random_state=333)
all_ewe_df = all_ewe_df.sample(frac=1.0, random_state=333)

all_fon_df.to_csv('Train_Fon_JW300.csv', index=False, encoding='utf-8')
all_ewe_df.to_csv('Train_Ewe_JW300.csv', index=False, encoding='utf-8')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

