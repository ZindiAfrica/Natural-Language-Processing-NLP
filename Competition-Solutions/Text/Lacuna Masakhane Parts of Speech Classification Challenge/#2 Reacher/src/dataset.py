from torch.utils.data import Dataset
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer,labels_to_ids, max_len, get_wids, istrain=False):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation
        self.istrain = istrain
        
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.labels_to_ids = labels_to_ids

    
    def __getitem__(self, index):
        LABEL_ALL_SUBTOKENS = True
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.target[index] if not self.get_wids else None
    
        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                             is_split_into_words=True,
                             #return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:              
                    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids
    
        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)
        
        # Masking 10% of tokens as Augmentation
        if not self.get_wids  and self.istrain and np.random.uniform()<0.3:
            ix = torch.rand(size=(len(item['input_ids']),)) < 0.12
            #ids = torch.tensor(input_ids, dtype=torch.long)
            # print(item['input_ids'].shape)
            # print(ix.shape)
            item['input_ids'][ix] = self.mask_token
        
        return item
    
    def __len__(self):
        return self.len